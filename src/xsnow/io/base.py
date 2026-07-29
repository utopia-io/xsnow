# io/base.py
"""
Shared constants, mappings, and helper functions for xsnow I/O operations.

This module contains the common utilities used across all format-specific
I/O modules (pro, smet, netcdf, zarr).
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import re
import logging
from pathlib import Path
from typing import Dict, Any, IO, Optional, List, Union
from collections import Counter

import numpy as np
import pandas as pd

from ..registry import registry
from ..selectors import (
    TimeSelector, LocationSelector, SlopeSelector, RealizationSelector,
    TimeLike, LocationLike, SlopeLike, RealizationLike
)
from ..utils import get_logger

from .memory import estimate_decoded_nbytes


_ARCHIVE_AUTO_EAGER_BUDGET_FRACTION = 0.5
_ARCHIVE_AUTO_EAGER_CAP_BYTES = 4 * 1024**3
_ARCHIVE_AUTO_EAGER_FALLBACK_BYTES = 1024**3


def _resolve_archive_auto_eager_max_bytes(
    memory_budget_bytes: Optional[int],
) -> int:
    """Return the adaptive decoded-size ceiling for automatic eager reads."""
    if memory_budget_bytes is None:
        return _ARCHIVE_AUTO_EAGER_FALLBACK_BYTES

    budget_based_limit = max(
        1,
        int(memory_budget_bytes * _ARCHIVE_AUTO_EAGER_BUDGET_FRACTION),
    )
    return min(budget_based_limit, _ARCHIVE_AUTO_EAGER_CAP_BYTES)


def _resolve_archive_lazy(
    ds: Any,
    lazy: Optional[bool],
    memory_budget_bytes: Optional[int] = None,
) -> bool:
    """Resolve archive auto mode from decoded size and the memory budget."""
    if lazy is not None:
        return lazy

    # A size that cannot be trusted (object or variable-length string data)
    # keeps the archive lazy rather than gambling on an eager load.
    estimated_nbytes = estimate_decoded_nbytes(ds)
    if estimated_nbytes is None:
        return True

    auto_eager_max_bytes = _resolve_archive_auto_eager_max_bytes(
        memory_budget_bytes
    )
    return estimated_nbytes >= auto_eager_max_bytes


# --- Dynamic Mapping from Registry ---
CODE_TO_NAME_MAP: Dict[str, str] = {
    details['param_code']: name
    for name, details in registry.items() if 'param_code' in details
}
CODE_TO_NAME_MAP["0500"] = "timestamp"

# --- Static Mappings for .pro file headers ---
HEADER_MAP: Dict[str, str] = {
    'StationName': 'station_name',
    'Altitude=': 'altitude',
    'Latitude=': 'latitude',
    'Longitude=': 'longitude',
    'SlopeAngle=': 'slopeAngle',
    'SlopeAzi=': 'slopeAzi',
    'StationName=': 'stationName',
    'tz ': 'tz',        # Format: "tz = 1" (space before =)
    'tz=': 'tz',        # Format: "tz=1" (no space)
    'TZ=': 'tz',
    'TimeZone=': 'tz',
    'timezone=': 'tz',
}

PRO_FILE_REGEX = re.compile(r".*\.pro(|\.gz|\.bz2|\.xz)$")
SMET_FILE_REGEX = re.compile(r".*\.smet(|\.gz|\.bz2|\.xz)$")


def _ext_is(filename: Union[str, Path], ext: str) -> bool:
    """Determine extension for str or Path."""
    if isinstance(filename, str):
        return filename.endswith(ext)
    elif isinstance(filename, Path):
        return filename.suffix == ext
    else:
        raise ValueError(f"Unsupported filename type: {type(filename)}")


def _open_any_as_text(filename: Union[str, Path], mode="rt", encoding="utf-8", **kwargs) -> IO:
    """Open an uncompressed or compressed file in text mode.

    For compatibility across different ``open`` implementations, ``mode`` is
    always coerced to text mode.
    """
    if "t" not in mode:
        mode += "t"
    if _ext_is(filename, ".gz"):
        return gzip.open(filename, mode, encoding=encoding, **kwargs)
    elif _ext_is(filename, ".bz2"):
        return bz2.open(filename, mode, encoding=encoding, **kwargs)
    elif _ext_is(filename, ".xz"):
        return lzma.open(filename, mode, encoding=encoding, **kwargs)
    else:
        return open(filename, mode, encoding=encoding, **kwargs)


def _round_coord(val: Any, ndigits: int = 6) -> float:
    """Round coordinate-like value to `ndigits`, preserving NaN."""
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(fval):
        return np.nan
    return round(fval, ndigits)


def _sanitize_station_name(name: str) -> str:
    """
    Sanitize station name for use as xarray coordinate.
    
    Rules:
    - Strip leading/trailing whitespace
    - Replace whitespace sequences with single underscore
    - Replace problematic characters with underscore
    - Ensure non-empty
    - Limit length to 64 characters
    
    Parameters
    ----------
    name : str
        Raw station name from .pro file
    
    Returns
    -------
    str
        Sanitized station name safe for xarray/NetCDF
    
    Examples
    --------
    >>> _sanitize_station_name("Station A-1")
    'Station_A-1'
    >>> _sanitize_station_name("  VIR1A  ")
    'VIR1A'
    >>> _sanitize_station_name("")
    'UNNAMED_STATION'
    """
    # Strip whitespace
    name = name.strip()
    
    # If empty, return generic name
    if not name:
        return "UNNAMED_STATION"
    
    # Replace sequences of whitespace with single underscore
    name = re.sub(r'\s+', '_', name)
    
    # Replace problematic characters (keep alphanumeric, underscore, hyphen, period)
    # This ensures NetCDF compatibility
    name = re.sub(r'[^\w\-.]', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Limit length
    MAX_NAME_LENGTH = 64
    if len(name) > MAX_NAME_LENGTH:
        name = name[:MAX_NAME_LENGTH]
    
    return name


def _handle_duplicate_names(
    names: List[str], 
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Ensure all station names are unique by appending indices to duplicates.
    
    Parameters
    ----------
    names : list[str]
        Original station names (may contain duplicates)
    logger : logging.Logger, optional
        Logger instance for warnings
    
    Returns
    -------
    list[str]
        Unique station names with indices appended to duplicates
    
    Examples
    --------
    >>> _handle_duplicate_names(['A', 'B', 'A', 'C', 'A'])
    ['A_0', 'B', 'A_1', 'C', 'A_2']
    >>> _handle_duplicate_names(['A', 'B', 'C'])
    ['A', 'B', 'C']
    """
    if len(names) == len(set(names)):
        return names
    
    log = get_logger(__name__, logger=logger)
    
    # Find duplicates
    name_counts = Counter(names)
    duplicates = {name for name, count in name_counts.items() if count > 1}
    
    # Rename all occurrences of duplicate names
    occurrence_counter: Dict[str, int] = {}
    unique_names: List[str] = []
    warned: set = set()
    
    for name in names:
        if name in duplicates:
            count = occurrence_counter.get(name, 0)
            unique_name = f"{name}_{count}"
            occurrence_counter[name] = count + 1
            
            # Warn once per duplicate name
            if name not in warned:
                log.warning(
                    f"Duplicate station name '{name}' found. "
                    f"Renaming all occurrences with indices (e.g., {name}_0, {name}_1, ...)."
                )
                warned.add(name)
        else:
            unique_name = name
        
        unique_names.append(unique_name)
    
    return unique_names


# --- Selector Coercion Functions ---

def _coerce_time(sel: Optional[TimeLike]) -> Optional[TimeSelector]:
    """Coerce time-like input to TimeSelector."""
    if sel is None or isinstance(sel, TimeSelector):
        return sel
    return TimeSelector(sel=sel)


def _coerce_location(sel: Optional[LocationLike]) -> Optional[LocationSelector]:
    """Coerce location-like input to LocationSelector."""
    if sel is None or isinstance(sel, LocationSelector):
        return sel
    return LocationSelector(sel=sel)


def _coerce_slope(sel: Optional[SlopeLike]) -> Optional[SlopeSelector]:
    """Coerce slope-like input to SlopeSelector."""
    if sel is None or isinstance(sel, SlopeSelector):
        return sel
    return SlopeSelector(sel=sel)


def _coerce_realization(sel: Optional[RealizationLike]) -> Optional[RealizationSelector]:
    """Coerce realization-like input to RealizationSelector."""
    if sel is None or isinstance(sel, RealizationSelector):
        return sel
    return RealizationSelector(sel=sel) 
