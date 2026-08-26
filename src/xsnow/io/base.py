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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, IO, Literal, Optional, List, Sequence, Union
from collections import Counter

import dask
import numpy as np
import pandas as pd
import xarray as xr

from ..registry import registry
from ..selectors import (
    TimeSelector, LocationSelector, SlopeSelector, RealizationSelector,
    TimeLike, LocationLike, SlopeLike, RealizationLike
)
from ..utils import get_logger, normalize_slope_coords

from .memory import _check_estimated_memory_budget, estimate_decoded_nbytes


_ARCHIVE_AUTO_EAGER_BUDGET_FRACTION = 0.5
_ARCHIVE_AUTO_EAGER_CAP_BYTES = 4 * 1024**3
_ARCHIVE_AUTO_EAGER_FALLBACK_BYTES = 1024**3
_EAGER_CAST_CHUNK_BYTES = 32 * 1024**2


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


def _promote_slope_metadata_coords(
    ds: xr.Dataset,
    slope_coord_names: Sequence[str] = ("azimuth", "inclination"),
) -> xr.Dataset:
    """
    Promote legacy slope metadata data variables back to coordinates.

    This preserves backwards compatibility with files written while
    ``azimuth`` or ``inclination`` were saved as data variables. Only
    variables carrying a ``slope`` dimension are promoted. Coordinate
    dimensions are not normalized here; callers apply their requested
    ``norm_slopes`` policy separately.
    """
    to_promote = [
        name
        for name in slope_coord_names
        if name in ds.data_vars and "slope" in ds[name].dims
    ]
    if not to_promote:
        return ds

    return ds.set_coords(to_promote)


def _apply_norm_slopes(
    ds: xr.Dataset,
    norm_slopes: Union[bool, Literal["auto"]] = "auto",
    slope_coord_names: Sequence[str] = ("azimuth", "inclination"),
    logger: Optional[logging.Logger] = None,
) -> xr.Dataset:
    """Apply the public ``norm_slopes`` read policy to recognized coordinates."""
    _validate_norm_slopes(norm_slopes)

    if norm_slopes is False or "slope" not in ds.dims:
        return ds

    if norm_slopes is True:
        return normalize_slope_coords(
            ds,
            slope_coord_names=slope_coord_names,
            strategy="union",
            logger=logger,
        )

    return normalize_slope_coords(
        ds,
        slope_coord_names=slope_coord_names,
        strategy="auto",
        logger=logger,
    )


def _validate_norm_slopes(
    norm_slopes: Union[bool, Literal["auto"]],
) -> None:
    """Validate the public slope-normalization read mode."""
    if (
        norm_slopes is not True
        and norm_slopes is not False
        and norm_slopes != "auto"
    ):
        raise ValueError(
            f"Unsupported norm_slopes value: {norm_slopes!r}. "
            "Use True, False, or 'auto'."
        )


FloatDtypeLike = Union[str, np.dtype, None]


def _validate_float_dtype(float_dtype: FloatDtypeLike) -> Optional[np.dtype]:
    """Validate the public ``float_dtype`` read policy value."""
    if float_dtype is None:
        return None
    try:
        target = np.dtype(float_dtype)
    except TypeError:
        target = None
    if target is None or target.kind != "f":
        raise ValueError(
            f"Unsupported float_dtype value: {float_dtype!r}. "
            "Use a floating-point dtype such as 'float32', 'float64', or None."
        )
    return target


def _apply_float_dtype(
    ds: xr.Dataset,
    float_dtype: FloatDtypeLike,
) -> xr.Dataset:
    """Cast float variables and coordinates to ``float_dtype``.

    ``None`` preserves source dtypes. Dask-backed arrays stay lazy.
    """
    target = _validate_float_dtype(float_dtype)
    if target is None:
        return ds

    data_casts = {
        name: ds[name].astype(target)
        for name, var in ds.data_vars.items()
        if var.dtype.kind == "f" and var.dtype != target
    }
    coord_casts = {
        name: ds[name].astype(target)
        for name, var in ds.coords.items()
        if var.dtype.kind == "f" and var.dtype != target
    }
    if not data_casts and not coord_casts:
        return ds

    ds = ds.copy()
    for name, cast in data_casts.items():
        ds[name] = cast
    if coord_casts:
        ds = ds.assign_coords(coord_casts)
    return ds


@dataclass(frozen=True)
class _FloatDtypePlan:
    """Metadata-only plan for applying a floating-point dtype policy."""

    estimated_nbytes: Optional[int]
    largest_source_cast_nbytes: int
    requires_cast: bool


def _plan_float_dtype(
    ds: xr.Dataset,
    float_dtype: FloatDtypeLike,
) -> _FloatDtypePlan:
    """Estimate the post-cast payload and largest source cast allocation."""
    target = _validate_float_dtype(float_dtype)
    estimated_nbytes = 0
    estimate_unknown = False
    largest_source_cast_nbytes = 0
    requires_cast = False

    try:
        for variable in ds.variables.values():
            source_dtype = np.dtype(variable.dtype)
            if source_dtype.kind in {"O", "T"}:
                estimate_unknown = True
                continue

            variable_size = int(variable.size)
            cast_variable = (
                target is not None
                and source_dtype.kind == "f"
                and source_dtype != target
            )
            if cast_variable:
                requires_cast = True
                largest_source_cast_nbytes = max(
                    largest_source_cast_nbytes,
                    variable_size * source_dtype.itemsize,
                )
                effective_dtype = target
            else:
                effective_dtype = source_dtype

            estimated_nbytes += variable_size * effective_dtype.itemsize
    except (AttributeError, TypeError, ValueError, OverflowError):
        estimate_unknown = True

    return _FloatDtypePlan(
        estimated_nbytes=None if estimate_unknown else estimated_nbytes,
        largest_source_cast_nbytes=largest_source_cast_nbytes,
        requires_cast=requires_cast,
    )


def _apply_float_dtype_eager_in_place(
    ds: xr.Dataset,
    float_dtype: FloatDtypeLike,
) -> xr.Dataset:
    """Cast one file-backed variable at a time and release its source data."""
    target = _validate_float_dtype(float_dtype)
    if target is None:
        return ds

    for name in list(ds.data_vars):
        variable = ds[name].variable
        if variable.dtype.kind == "f" and variable.dtype != target:
            ds[name] = variable.astype(target)

    for name in list(ds.coords):
        variable = ds.coords[name].variable
        if variable.dtype.kind == "f" and variable.dtype != target:
            ds.coords[name] = variable.astype(target)

    return ds


def _materialize_eager_archive(
    ds: xr.Dataset,
    float_dtype: FloatDtypeLike,
    *,
    memory_budget_bytes: Optional[int],
    source: str,
    logger: Optional[logging.Logger] = None,
    cast_chunk_bytes: int = _EAGER_CAST_CHUNK_BYTES,
) -> xr.Dataset:
    """Apply an archive dtype policy safely and return in-memory NumPy data.

    The post-cast payload is checked before any cast allocation. Small archive
    variables use xarray's fast non-Dask backend path and are converted before
    the remaining variables are loaded. A large variable is wrapped in bounded
    Dask chunks and computed sequentially so conversion does not require the
    complete source and target arrays in memory at the same time.
    """
    log = get_logger(__name__, logger=logger)
    plan = _plan_float_dtype(ds, float_dtype)
    _check_estimated_memory_budget(
        plan.estimated_nbytes,
        memory_budget_bytes,
        raise_on_exceed=True,
        raise_on_unknown=True,
        source=source,
        logger=log,
    )

    use_chunked_cast = (
        plan.requires_cast
        and plan.largest_source_cast_nbytes > cast_chunk_bytes
    )
    if use_chunked_cast:
        log.debug(
            "Using bounded chunks for eager archive dtype conversion: "
            f"largest source variable is {plan.largest_source_cast_nbytes:,} "
            f"bytes; chunk target is {cast_chunk_bytes:,} bytes."
        )
        with dask.config.set({"array.chunk-size": f"{cast_chunk_bytes}B"}):
            ds = ds.chunk("auto")

    if use_chunked_cast:
        ds = _apply_float_dtype(ds, float_dtype)
        ds.load(scheduler="single-threaded")
    else:
        ds = _apply_float_dtype_eager_in_place(ds, float_dtype)
        ds.load()
    return ds


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


def _round_coord(val: Any, ndigits: int = 4) -> float:
    """Round coordinate-like value to `ndigits`, preserving NaN.

    Four decimals (~11 m) is the finest grid float32 storage resolves
    unambiguously at every coordinate: the worst spacing is 1.5e-5 near
    180 against a 5e-5 half grid step. It is also the granularity the
    platform already identifies stations by, through the ``{lat:.4f}``
    station ids adatahub builds.
    """
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
