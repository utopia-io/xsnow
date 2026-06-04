# xsnow_io.py
"""
Backward compatibility facade for xsnow I/O operations.

This module re-exports the public I/O API from the io subpackage.
All implementation has been refactored into:
- io/base.py     - Shared constants and helper functions
- io/pro.py      - SNOWPACK .pro file parsing and writing
- io/smet.py     - SMET file parsing and writing
- io/netcdf.py   - NetCDF read/write operations
- io/zarr.py     - Zarr read/write operations
- io/merge.py    - Dataset merging and file resolution
- io/__init__.py - Main read() dispatcher

For new code, prefer importing directly from xsnow:
    from xsnow import read, to_netcdf, to_zarr, to_pro, to_smet

Or from the io subpackage for internal functions:
    from xsnow.io import read
    from xsnow.io.pro import _read_pro_raw
"""

# Re-export public API from io subpackage
from .io import (
    read,
    read_smet,
    to_pro,
    to_smet,
    to_netcdf,
    to_zarr,
)

# Re-export internal functions for backward compatibility with tests
from .io.base import (
    CODE_TO_NAME_MAP,
    HEADER_MAP,
    _round_coord,
    _sanitize_station_name,
    _handle_duplicate_names,
    _coerce_time,
    _coerce_location,
    _coerce_slope,
    _coerce_realization,
)

from .io.pro import (
    _read_pro_raw,
    _read_pro_metadata,
    _create_pro_dataset,
    _scan_pro_for_max_layers,
    _get_layer_count,
    _read_pro_files_eager,
    _read_pro_files_lazy,
    _extract_metadata_fast,
    _determine_max_layers_smart,
    _validate_files_for_smart_detection,
)

from .io.smet import (
    _read_smet_files,
)

from .io.netcdf import (
    _read_xsnow_netcdf,
    _encode_mapping_attrs_for_netcdf,
    _decode_mapping_attrs_from_netcdf,
)

from .io.zarr import (
    _read_zarr,
)

from .io.merge import (
    _resolve_filepaths,
    _merge_and_finalize,
)

__all__ = [
    'read',
    'read_smet',
    'to_pro',
    'to_smet',
    'to_netcdf',
    'to_zarr',
    'append_latest',
    'trim_padding',
    # Internal functions (for backward compatibility)
    'CODE_TO_NAME_MAP',
    'HEADER_MAP',
    '_round_coord',
    '_sanitize_station_name',
    '_handle_duplicate_names',
    '_read_pro_raw',
    '_read_pro_metadata',
    '_create_pro_dataset',
    '_scan_pro_for_max_layers',
    '_read_smet_files',
    '_resolve_filepaths',
    '_merge_and_finalize',
]


# =============================================================================
# TEMPORARY: append_latest and trim_padding 
# These functions could eventually be moved, but are kept here for backward compatibility.
# Possible new destinations:
# - trim_padding --> if useful for end user: utils; otherwise remain in io/
# - append_latest --> xsnow_io (currently), io/base or io/__init()__ (next to read())
# =============================================================================

import logging
from datetime import datetime
from pathlib import Path
from typing import cast, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .core import xsnowDataset
from .plugin import register_xsnowDataset_method
from .combiners import concat as xsnow_concat
from .utils import get_logger


@register_xsnowDataset_method
def append_latest(
    existing_ds: xsnowDataset, 
    source: Union[str, List[str], Path],
    from_time: Optional[Union[str, datetime, pd.Timestamp, np.datetime64]] = None,
    join: str = 'left', 
    logger: Optional[logging.Logger] = None, 
    **kwargs,
) -> xsnowDataset:
    """
    Incrementally extend a dataset by appending newer timestamps from files.

    This method is a convenience wrapper around time selection, `read()`, and
    `xsnow.concat()`. It is intended for cache/update workflows where an
    existing dataset should be extended with only the timestamps that are newer
    than the existing time coordinate.

    Semantics:
        1. Determine a read start time:
           - If `from_time` is None, use max(existing time) + 1 second.
           - If `from_time` is provided, keep existing times strictly before
             `from_time` and read new data starting at `from_time` inclusive.
        2. Read `source` with that time filter.
        3. Reject overlapping timestamps between the retained existing data
           and the newly read data.
        4. Concatenate along `time` using `xsnow.concat`.

    Args:
        existing_ds: Dataset instance to extend.
        source: Path, list of paths, or directory to pass to `read()` for new
            data.
        from_time: Optional explicit update boundary. Existing data at or
            after this timestamp is dropped before new data is appended.
            Accepts strings, Python datetimes, pandas timestamps, or numpy
            datetimes.
        join: Join mode for non-time dimensions forwarded to `xsnow.concat`.
            The default, 'left', keeps the existing non-time coordinate domain.
            Use 'outer' or 'right' when new non-time coordinates should be
            allowed into the result.
        logger: Optional logger for status and warning messages.
        **kwargs: Additional keyword arguments forwarded to `xsnow.concat`,
            such as `compat`, `coords`, or `combine_attrs`.

    Returns:
        An `xsnowDataset` containing the retained existing data plus newly
        appended timestamps. If no new time steps are found, returns a dataset
        built from the retained existing data.

    Raises:
        ValueError: If retained existing data and newly read data contain
            overlapping timestamps.

    Notes:
        The current implementation resets `azimuth`, `inclination`, and
        `aspect` from coordinates to data variables before concatenation to
        avoid coordinate/variable structure mismatches. This affects the
        returned dataset structure and should be considered part of the current
        behavior, not a statement about the desired long-term data model.
    """
    from .io import read
    from .selectors import TimeSelector
    
    logger = get_logger(__name__, logger=logger)

    base_data = existing_ds.data if existing_ds.data is not None else xr.Dataset()
    base_filtered = base_data
    base_max_ts = None
    if "time" in base_data and base_data.time.size > 0:
        base_max_ts = pd.to_datetime(base_data.time.max().values)

    time_start: Optional[str] = None
    if from_time is None:
        if "time" in base_data and base_data.time.size > 0:
            last_ts = pd.to_datetime(base_data.time.max().values)
            time_start = (last_ts + pd.Timedelta(seconds=1)).isoformat()
    else:
        start_ts = pd.to_datetime(from_time)
        time_start = start_ts.isoformat()
        if base_max_ts is not None and start_ts < base_max_ts:
            logger.warning(
                "append_latest called with from_time earlier than existing max time"
            )
        if "time" in base_data:
            mask = base_data["time"].values < start_ts
            base_filtered = base_data.sel(time=mask)

    time_sel = TimeSelector(sel=slice(time_start, None)) if time_start else None
    new_ds = read(source, time=time_sel, logger=logger)
    new_data = new_ds.data if new_ds is not None else None

    def _new_coord_values(base: xr.Dataset, incoming: xr.Dataset) -> Dict[str, np.ndarray]:
        new_coords: Dict[str, np.ndarray] = {}
        for coord_name, coord in incoming.coords.items():
            if coord_name == "time" or coord.ndim != 1:
                continue
            incoming_vals = np.asarray(coord.values).ravel()
            base_vals = np.asarray(base[coord_name].values).ravel() if coord_name in base else np.array([])
            incoming_vals = incoming_vals[~pd.isna(incoming_vals)]
            base_vals = base_vals[~pd.isna(base_vals)]
            diff = np.setdiff1d(incoming_vals, base_vals, assume_unique=True)
            if diff.size:
                new_coords[str(coord_name)] = diff
        return new_coords

    def _preview(vals: np.ndarray) -> str:
        as_list = list(np.atleast_1d(vals))
        shown = ", ".join(map(repr, as_list[:5]))
        suffix = "..." if len(as_list) > 5 else ""
        return f"[{shown}{suffix}]"

    if new_data is None or "time" not in new_data or new_data.time.size == 0:
        if new_data is not None and join in {"outer", "right"}:
            extra_coords = _new_coord_values(base_filtered, new_data)
            if extra_coords:
                preview = ", ".join(
                    f"{name}: {_preview(vals)}" for name, vals in extra_coords.items()
                )
                logger.warning(
                    f"append_latest found no new time steps, but new coordinate values exist: {preview}"
                )
        return existing_ds.__class__(base_filtered)

    if "time" in base_filtered and "time" in new_data:
        overlap = np.intersect1d(base_filtered["time"].values, new_data["time"].values)
        if overlap.size:
            raise ValueError(f"append_latest detected overlapping timestamps: {overlap[:5]!r}")

    JoinMode = Literal["left", "right", "outer", "inner", "union"]
    
    slope_coords = ['azimuth', 'inclination', 'aspect']
    for coord in slope_coords:
        if coord in base_filtered.coords:
            base_filtered = base_filtered.reset_coords(coord)
        if coord in new_data.coords:
            new_data = new_data.reset_coords(coord)
    
    combined = xsnow_concat(
        [existing_ds.__class__(base_filtered), xsnowDataset(new_data)],
        dim="time",
        join=cast(JoinMode, join),
        **kwargs,
    )
    
    new_times = combined.time.where(combined.time >= pd.to_datetime(time_start), drop=True)
    new_time_start = pd.to_datetime(new_times[0].values).isoformat()
    new_time_end = pd.to_datetime(new_times.max().values).isoformat()
    logger.info(f"Appended {len(new_times)} new timestamps: {new_time_start}--{new_time_end}")
    
    combined_data = combined.data if hasattr(combined, 'data') else combined
    extra_coords = _new_coord_values(base_filtered, combined_data)
    if extra_coords:
        preview = ", ".join(f"{name}: {_preview(vals)}" for name, vals in extra_coords.items())
        logger.debug(f"Added new coordinates: {preview}")

    return combined if isinstance(combined, xsnowDataset) else xsnowDataset(combined)


@register_xsnowDataset_method
def trim_padding(
    ds: xsnowDataset,
    dims: Optional[List[str]] = None,
    keep_buffer: int = 0,
    inplace: bool = False
) -> xsnowDataset:
    """
    Remove NaN padding from layer dimension to reduce memory usage.

    Loading .pro files with a fixed or auto-detected `max_layers` can create
    trailing layers that are NaN for every variable. This method finds the last
    index containing data along each requested dimension and trims all trailing
    all-NaN slices.

    Args:
        ds: Dataset instance to trim.
        dims: Dimensions to trim. Defaults to ['layer'].
        keep_buffer: Number of extra slices to keep after the last non-NaN
            slice. This is useful when a dataset will be appended to later.
            The result is never expanded beyond the original size.
        inplace: If True, modify and return `ds`. If False (default), trim a
            copy and leave the original unchanged.

    Returns:
        The trimmed `xsnowDataset`. This is the same object when
        `inplace=True`, otherwise a new wrapper around copied data.

    Examples:
        >>> # Dataset padded to 50 layers, but only 12 contain data
        >>> ds_trimmed = ds.trim_padding()
        >>> ds_trimmed.sizes["layer"]
        12

        >>> # Keep two empty layers as future append buffer
        >>> ds.trim_padding(keep_buffer=2)

        >>> # Modify the existing dataset object
        >>> ds.trim_padding(inplace=True)

    Notes:
        This method assumes padding is at the end of the dimension. It checks
        all data variables that contain each requested dimension. For Dask
        backed arrays, the `.values` access used to detect non-NaN slices
        triggers computation.
    """
    logger = logging.getLogger(__name__)
    
    if dims is None:
        dims = ['layer']
    
    if inplace:
        ds_work = ds
    else:
        ds_work = ds.copy()
    
    current_data = ds_work.data
    
    for dim in dims:
        if dim not in current_data.dims:
            logger.warning(f"Dimension '{dim}' not found in dataset, skipping")
            continue
        
        original_size = current_data.sizes[dim]
        max_valid_index = -1
        
        relevant_vars = [
            v for v in current_data.data_vars 
            if dim in current_data[v].dims
        ]
        
        if not relevant_vars:
            continue

        for var_name in relevant_vars:
            da_var = current_data[var_name]
            reduce_dims = [d for d in da_var.dims if d != dim]
            has_data = da_var.notnull().any(dim=reduce_dims)
            has_data_np = has_data.values
            
            if has_data_np.any():
                last_idx = np.where(has_data_np)[0][-1]
                max_valid_index = max(max_valid_index, last_idx)
        
        if max_valid_index == -1:
            new_size = 0
        else:
            new_size = max_valid_index + 1 + keep_buffer
            
        new_size = min(new_size, original_size)
        
        if new_size < original_size:
            current_data = current_data.isel({dim: slice(0, new_size)})
            logger.info(
                f"Trimmed '{dim}' dimension from {original_size} to {new_size} "
                f"(removed {original_size - new_size} NaN-only slices)"
            )
        else:
            logger.debug(f"No trimming needed for dimension '{dim}'")

    ds_work.data = current_data
    return ds_work
