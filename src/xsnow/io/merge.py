# io/merge.py
"""
Dataset merging and file resolution utilities for xsnow I/O.

This module contains functions for:
- Resolving file paths from various source types
- Merging multiple datasets into a unified xsnowDataset
- Applying final filters and coordinate normalization
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..core import xsnowDataset
from ..registry import registry
from ..selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from ..utils import get_logger, normalize_slope_coords

from .base import _handle_duplicate_names


def _resolve_filepaths(
    source: Union[str, Path, Sequence[Union[str, Path]]],
    recursive: bool = False, 
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Resolves the source argument into separate lists of .pro and .smet files.

    Args:
        source: The data source, which can be a directory path, a single
            file path, or a list of paths.
        recursive: If True, search subdirectories recursively.
        logger: Optional logger instance.

    Returns:
        A tuple containing two lists: one for .pro file paths and one for
        .smet file paths.
        
    Raises:
        TypeError: If the source is of an unsupported type.
    """
    logger = logger or get_logger(__name__)
    paths_to_process: List[Path] = []
    
    if isinstance(source, (str, Path)):
        paths_to_process.append(Path(source))
    elif isinstance(source, (list, tuple)):
        paths_to_process = [Path(p) for p in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    resolved_files: List[Path] = []
    for p in paths_to_process:
        if not p.exists():
            logger.warning(f"Source path does not exist and will be skipped: {p}")
            continue
        
        if p.is_dir():
            pattern = '**/*.pro' if recursive else '*.pro'
            resolved_files.extend(p.glob(pattern))
            pattern = '**/*.smet' if recursive else '*.smet'
            resolved_files.extend(p.glob(pattern))
        else:
            resolved_files.append(p)

    unique_files = sorted(list(dict.fromkeys(resolved_files)))

    pro_files = [f for f in unique_files if f.suffix == '.pro']
    smet_files = [f for f in unique_files if f.suffix == '.smet']
    
    return pro_files, smet_files


def _merge_and_finalize(
    all_datasets: List[xr.Dataset],
    time: Optional[TimeSelector],
    location: Optional[LocationSelector],
    slope: Optional[SlopeSelector],
    realization: Optional[RealizationSelector],
    logger: logging.Logger,
    norm_slopes: Union[bool, Literal['auto']] = 'auto',
) -> Optional[xsnowDataset]:
    """
    Merges, re-indexes, and applies final filters to the datasets.

    This function combines a list of xarray.Dataset objects into a single,
    unified xsnowDataset. It handles re-indexing of the 'location'
    dimension from station names to integers, applies any specified filters,
    and computes the final z-depth coordinate.

    Args:
        all_datasets: A list of xarray.Dataset objects to be merged.
        time: A TimeSelector for final time-based filtering.
        location: A LocationSelector for final location-based filtering.
        slope: A SlopeSelector for final slope-based filtering.
        realization: A RealizationSelector for final realization-based filtering.
        logger: Logger instance.
        norm_slopes: Controls slope normalization:
            - 'auto' (default): normalize when slopes are consistent across locations;
              preserve per-location slopes when inconsistent.
            - True: force normalization to `(slope,)` even if values differ by location.
            - False: do not call `normalize_slope_coords`.

    Returns:
        A unified and filtered xsnowDataset, or None if no datasets were provided.
    """
    logger = get_logger(__name__, logger=logger)
    
    if not all_datasets:
        return None

    # Enforce timezone consistency
    final_timezone = None
    for ds in all_datasets:
        if 'time' in ds and 'timezone' in ds.time.attrs:
            current_tz = ds.time.attrs['timezone']
            if final_timezone is None:
                final_timezone = current_tz
            elif final_timezone != current_tz:
                raise ValueError(
                    f"Conflicting timezones: '{final_timezone}' vs '{current_tz}'.")

    # Group datasets by location (location is already a string)
    grouped_by_location: Dict[str, List[xr.Dataset]] = {}
    for ds in all_datasets:
        loc = str(ds.location.values[0])  # Get string location name
        grouped_by_location.setdefault(loc, []).append(ds)
    
    # HANDLE DUPLICATE STATION NAMES
    station_names = list(grouped_by_location.keys())
    unique_names = _handle_duplicate_names(station_names, logger=logger)
    
    # Rename locations if duplicates were found
    if unique_names != station_names:
        new_grouped: Dict[str, List[xr.Dataset]] = {}
        for old_name, new_name in zip(station_names, unique_names):
            new_grouped[new_name] = grouped_by_location[old_name]
            # Update location coordinate in datasets
            for ds in new_grouped[new_name]:
                ds['location'] = [new_name]
        grouped_by_location = new_grouped
    
    merged_datasets: List[xr.Dataset] = []
    for loc_name, dsets in grouped_by_location.items():    
        profile_dsets = [ds for ds in dsets if 'layer' in ds.dims]
        timeseries_dsets = [ds for ds in dsets if 'layer' not in ds.dims]

        combined_profiles_ds = None
        if profile_dsets:
            if len(profile_dsets) > 1:
                # Sort by inclination if it exists, otherwise use 0 as default
                def get_slope_key(ds: xr.Dataset) -> float:
                    if 'inclination' in ds.coords:
                        return float(ds.coords['inclination'].values[0])
                    elif 'inclination' in ds.data_vars:
                        return float(ds['inclination'].values[0])
                    else:
                        return 0.0  # Default value if inclination doesn't exist
                
                profile_dsets.sort(key=get_slope_key)
                
                # CRITICAL FIX: Save location-level coordinates before slope concat
                # These coordinates (lat/lon/alt) should NOT vary with slope dimension
                location_coords: Dict[str, xr.DataArray] = {}
                for coord_name in ['latitude', 'longitude', 'altitude', 'location']:
                    if coord_name in profile_dsets[0].coords:
                        location_coords[coord_name] = profile_dsets[0][coord_name]
                
                # Drop location-level coords before concatenating along slope
                profile_dsets_cleaned = [
                    ds.drop_vars(
                        [c for c in ['latitude', 'longitude', 'altitude', 'location'] if c in ds.coords],
                        errors='ignore'
                    )
                    for ds in profile_dsets
                ]
                
                # Extract underlying xarray Datasets before concat
                profile_xr_datasets = [
                    ds.data if hasattr(ds, 'data') else ds 
                    for ds in profile_dsets_cleaned
                ]
                combined_profiles_ds = xr.concat(
                    profile_xr_datasets, dim='slope', join='outer', coords='all'
                )
                combined_profiles_ds = combined_profiles_ds.assign_coords(
                    slope=np.arange(combined_profiles_ds.sizes['slope'])
                )
                
                # Restore location-level coordinates (without slope dimension)
                combined_profiles_ds = combined_profiles_ds.assign_coords(location_coords)
            else:
                # Extract .data if it's an xsnowDataset
                combined_profiles_ds = (
                    profile_dsets[0].data if hasattr(profile_dsets[0], 'data') 
                    else profile_dsets[0]
                )

        # Extract .data from all datasets before merge
        all_to_merge: List[xr.Dataset] = []
        if combined_profiles_ds is not None:
            all_to_merge.append(combined_profiles_ds)
        # Also extract .data from timeseries datasets
        for ds in timeseries_dsets:
            all_to_merge.append(ds.data if hasattr(ds, 'data') else ds)
        
        if not all_to_merge: 
            continue

        if len(all_to_merge) > 1:
            merged_loc_ds = xr.merge(
                all_to_merge, compat='override', join='outer', combine_attrs='no_conflicts'
            )
            merged_datasets.append(merged_loc_ds)
        else:
            merged_datasets.append(all_to_merge[0])

    if not merged_datasets: 
        return None

    # Sort by station name for consistent ordering
    merged_datasets.sort(key=lambda ds: str(ds.location.values[0]))
    
    # Extract .data from merged datasets (xr.concat needs plain xarray Datasets)
    merged_xr_datasets = [ds.data if hasattr(ds, 'data') else ds for ds in merged_datasets]
    final_xr_dataset = xr.concat(
        merged_xr_datasets,
        dim='location', 
        join='outer', 
        coords='all',
        compat='override'
    )
    
    # Preserve CF-compliant metadata
    if not final_xr_dataset.attrs:
        final_xr_dataset.attrs = {
            'Conventions': 'CF-1.8',
            'crs': 'EPSG:4326'
        }

    # Add variable metadata
    for var_name_hashable in list(final_xr_dataset.variables):
        var_name = str(var_name_hashable)
        if reg_entry := registry.get(var_name):
            final_xr_dataset[var_name].attrs['standard_name'] = reg_entry.get('standard_name', 'unknown')
            final_xr_dataset[var_name].attrs['units'] = reg_entry.get('si_units', 'unknown')
            final_xr_dataset[var_name].attrs['long_name'] = reg_entry.get('standard_name', var_name).replace('_', ' ')
        else:
            # Set defaults for variables not in registry if not already set
            if var_name in final_xr_dataset.data_vars:
                if 'units' not in final_xr_dataset[var_name].attrs:
                    final_xr_dataset[var_name].attrs['units'] = '1'
                if 'standard_name' not in final_xr_dataset[var_name].attrs:
                    final_xr_dataset[var_name].attrs['standard_name'] = var_name
                if 'long_name' not in final_xr_dataset[var_name].attrs:
                    final_xr_dataset[var_name].attrs['long_name'] = var_name.replace('_', ' ')

    # Normalize slope coordinates
    if 'slope' in final_xr_dataset.dims:
        try:
            if norm_slopes is True:
                final_xr_dataset = normalize_slope_coords(
                    final_xr_dataset,
                    verify_consistency=False,
                    allow_per_location=False,
                )
            elif norm_slopes == 'auto':
                final_xr_dataset = normalize_slope_coords(
                    final_xr_dataset,
                    verify_consistency=True,
                    allow_per_location=True,
                )
            elif norm_slopes is False:
                pass  # Explicitly skip normalization
            else:
                raise ValueError(
                    f"Unsupported norm_slopes value: {norm_slopes!r}. "
                    "Use True, False, or 'auto'."
                )
        except ValueError as e:
            logger.error(
                f"Failed to normalize slope coordinates: {e}\n"
                f"This indicates a data structure issue that needs investigation."
            )
            raise
    
    # Apply final filters using selectors
    if time:
        final_xr_dataset = time.apply(final_xr_dataset)
    if location:
        final_xr_dataset = location.apply(final_xr_dataset)
    if slope:
        final_xr_dataset = slope.apply(final_xr_dataset)
    if realization:
        final_xr_dataset = realization.apply(final_xr_dataset)
    
    from ..utils import reorder_dims
    final_xr_dataset = reorder_dims(final_xr_dataset)
    final_dataset = xsnowDataset(final_xr_dataset)
    
    # Compute z-depth if not present
    if 'z' not in final_dataset and 'height' in final_dataset:
        final_dataset = final_dataset.compute_z_depth()
    
    # Compute HS if not present
    if 'HS' not in final_dataset and 'height' in final_dataset:
        final_dataset = final_dataset.compute_hs()
    
    # Update profile status
    final_dataset = final_dataset.update_profile_status()
    
    return final_dataset