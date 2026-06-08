#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zarr Utilities for SNOWPACK Archiving

Shared functionality between operational_archive_zarr.py and snowpack_archiver_zarr.py
"""

import xarray as xr
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# FILL VALUES AND ENCODING
# =============================================================================

FILL_VALUES = {
    'int8': -128,      # Closest to -999 that fits
    'int16': -999,     # SNOWPACK standard
    'int32': -999,     # SNOWPACK standard
    'int64': -999,     # SNOWPACK standard
    'uint8': 255,
    'uint16': 65535,
    'uint32': 4294967295,
    'uint64': 18446744073709551615,
}


def get_zarr_encoding(ds):
    """
    Get Zarr encoding with fill values for integer types.
    
    Uses SNOWPACK standard (-999) when possible, dtype-specific otherwise.
    
    Args:
        ds: xarray.Dataset
        
    Returns:
        dict: Encoding dictionary for to_zarr()
    """
    encoding = {}
    for var in ds.data_vars:
        dtype_str = str(ds[var].dtype)
        if 'int' in dtype_str:
            fill_value = FILL_VALUES.get(dtype_str, -999)
            encoding[var] = {'_FillValue': fill_value}
    
    return encoding


def save_to_zarr(ds, output_path, mode='w'):
    """
    Save dataset to Zarr with proper encoding.
    
    Handles fill values for integer types.
    Zarr v3 applies default compression automatically.
    
    Args:
        ds: xarray.Dataset
        output_path: Path or str for output
        mode: 'w' for write, 'a' for append
    """
    encoding = get_zarr_encoding(ds)
    
    if encoding:
        logger.debug(f"Saving with fill values for {len(encoding)} integer variables")
        ds.to_zarr(output_path, mode=mode, encoding=encoding)
    else:
        ds.to_zarr(output_path, mode=mode)


# =============================================================================
# CHUNKING
# =============================================================================

def get_optimal_chunks(ds):
    """
    Get optimal chunk sizes based on dataset dimensions.
    
    Only chunks dimensions that exist (handles both profile and time-series data).
    
    Args:
        ds: xarray.Dataset
        
    Returns:
        dict: Chunk sizes for each dimension
    """
    desired_chunks = {
        'location': 200,
        'time': 100,
        'layer': -1,
        'slope': -1,
        'realization': -1
    }
    
    chunks = {}
    for dim, chunk_size in desired_chunks.items():
        if dim in ds.dims:
            chunks[dim] = chunk_size
    
    logger.debug(f"Chunking with: {chunks}")
    return chunks


# =============================================================================
# PREPROCESSING FOR MERGE
# =============================================================================

def preprocess_for_merge(ds):
    """
    Preprocess datasets for merging.
    
    1. Converts slope coordinates to data variables
    2. Drops xsnow-derived and non-essential coordinates
    
    Args:
        ds: xarray.Dataset
        
    Returns:
        xarray.Dataset: Preprocessed dataset
    """
    # Always convert slope coordinates to data variables
    slope_coords = ['azimuth', 'inclination', 'aspect']
    
    for coord in slope_coords:
        if coord in ds.coords:
            ds = ds.reset_coords(coord)
            
            # Broadcast 1D to 2D if needed
            coord_var = ds[coord]
            if coord_var.dims == ('slope',) and 'location' in ds.dims and 'slope' in ds.dims:
                template = None
                for var_name in ds.data_vars:
                    if var_name == coord:
                        continue
                    var = ds[var_name]
                    if 'location' in var.dims and 'slope' in var.dims:
                        template = var
                        for dim in var.dims:
                            if dim not in ['location', 'slope']:
                                template = template.isel({dim: 0})
                        break
                
                if template is not None:
                    ds[coord] = ds[coord].broadcast_like(template)
    
    # Drop derived and non-essential coordinates
    essential_coords = [
        'location', 'time', 'layer', 'slope', 'realization',
        'latitude', 'longitude', 'altitude', 'elevation',
        'lat', 'lon'
    ]
    
    coords_to_drop = [c for c in ds.coords if c not in essential_coords]
    if coords_to_drop:
        logger.debug(f"Dropping coordinates: {coords_to_drop}")
        ds = ds.drop_vars(coords_to_drop, errors='ignore')
    
    return ds


# =============================================================================
# FILE COLLECTION
# =============================================================================

def find_snowpack_files(data_dir, pattern='*.pro'):
    """
    Find SNOWPACK files recursively.
    
    Args:
        data_dir: Path to search
        pattern: File pattern (*.pro, *.smet, or both)
        
    Returns:
        list: List of Path objects
    """
    data_dir = Path(data_dir)
    
    if pattern == 'both':
        files = list(data_dir.rglob('*.pro')) + list(data_dir.rglob('*.smet'))
    else:
        files = list(data_dir.rglob(pattern))
    
    return sorted(files)


def filter_files_by_mtime(files, min_age_seconds=0, max_age_seconds=None):
    """
    Filter files by modification time.
    
    Args:
        files: List of Path objects
        min_age_seconds: Minimum file age (skip newer files)
        max_age_seconds: Maximum file age (skip older files)
        
    Returns:
        list: Filtered files
    """
    import time
    current_time = time.time()
    
    filtered = []
    for filepath in files:
        try:
            mtime = filepath.stat().st_mtime
            age = current_time - mtime
            
            if min_age_seconds > 0 and age < min_age_seconds:
                continue
                
            if max_age_seconds is not None and age > max_age_seconds:
                continue
                
            filtered.append(filepath)
        except (OSError, PermissionError):
            continue
    
    return filtered


# =============================================================================
# BATCH MERGING
# =============================================================================

def merge_zarr_batches(batch_files, concat_dim='location', preprocess=True):
    """
    Merge multiple Zarr batch files.
    
    Args:
        batch_files: List of paths to Zarr batch files
        concat_dim: Dimension to concatenate along
        preprocess: Whether to apply preprocessing
        
    Returns:
        xarray.Dataset: Merged dataset
    """
    logger.info(f"Merging {len(batch_files)} batches...")
    
    datasets = []
    for batch_file in batch_files:
        logger.debug(f"  Loading {Path(batch_file).name}")
        ds = xr.open_zarr(batch_file)
        if preprocess:
            ds = preprocess_for_merge(ds)
        datasets.append(ds)
    
    if len(datasets) == 1:
        merged = datasets[0]
    else:
        # Use coords='minimal' to handle dimension mismatches (.pro vs .smet)
        merged = xr.concat(datasets, dim=concat_dim, join='outer',
                          coords='minimal', data_vars='all')
    
    logger.info(f"  Merged: {len(merged[concat_dim]):,} {concat_dim}s")
    
    return merged


def remove_duplicate_times(ds):
    """
    Remove duplicate timesteps from dataset.
    
    Args:
        ds: xarray.Dataset with time dimension
        
    Returns:
        xarray.Dataset: Dataset with unique times
    """
    if 'time' not in ds.dims:
        return ds
    
    # Find unique times
    _, unique_indices = np.unique(ds.time.values, return_index=True)
    
    # Select only unique times
    ds_unique = ds.isel(time=sorted(unique_indices))
    
    removed = len(ds.time) - len(ds_unique.time)
    if removed > 0:
        logger.debug(f"Removed {removed} duplicate time(s)")
    
    return ds_unique


# =============================================================================
# BATCH CREATION
# =============================================================================

def create_file_batches(files, batch_size=2000):
    """
    Split file list into batches.
    
    Args:
        files: List of file paths
        batch_size: Files per batch
        
    Returns:
        list: List of file batches
    """
    return [files[i:i + batch_size] 
            for i in range(0, len(files), batch_size)]


# =============================================================================
# COORDINATE UTILITIES
# =============================================================================

def check_coordinate_consistency(datasets):
    """
    Check if coordinates are consistent across datasets.
    
    Args:
        datasets: List of xarray.Dataset objects
        
    Returns:
        dict: Report of coordinate mismatches
    """
    if not datasets:
        return {}
    
    all_coords = set()
    for ds in datasets:
        all_coords.update(ds.coords.keys())
    
    report = {}
    for coord in all_coords:
        present_in = sum(1 for ds in datasets if coord in ds.coords)
        if present_in != len(datasets):
            report[coord] = f"Present in {present_in}/{len(datasets)} datasets"
    
    return report