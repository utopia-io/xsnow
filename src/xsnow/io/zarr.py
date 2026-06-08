# io/zarr.py
"""
Zarr I/O operations for xsnow.

Zarr archives provide a directory-based storage format that offers:
- Better stability (pure Python, no HDF5 crashes)
- Lower memory usage (streaming reads)
- Cloud-friendly (works with S3, GCS)
- Same xarray interface as NetCDF
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal, Optional, Union

import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from ..utils import get_logger, _promote_slope_metadata_coords

# Define the allowed modes matching xarray/zarr requirements
ZarrModes = Literal['w', 'w-', 'a', 'a-', 'r+', 'r']


def _read_zarr(
    path: Union[str, Path],
    time_sel: Optional[TimeSelector] = None,
    location_sel: Optional[LocationSelector] = None,
    slope_sel: Optional[SlopeSelector] = None,
    realization_sel: Optional[RealizationSelector] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[xsnowDataset]:
    """
    Internal function to read a Zarr archive.
    
    Called by read() when it detects a .zarr file or directory.
    Similar to _read_xsnow_netcdf() for NetCDF files.
    
    Args:
        path: Path to the .zarr directory
        time_sel: TimeSelector for filtering by time (already coerced)
        location_sel: LocationSelector for filtering by location (already coerced)
        slope_sel: SlopeSelector for filtering by slope (already coerced)
        realization_sel: RealizationSelector for filtering by realization (already coerced)
        logger: Logger instance
        
    Returns:
        xsnowDataset with the loaded data, or None if loading fails
    """
    logger = get_logger(__name__, logger=logger)
    
    try:
        path = Path(path)
        
        # Verify it's a Zarr directory
        if not path.exists():
            raise FileNotFoundError(f"Zarr archive not found: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
            
        # Check for Zarr metadata (v2 or v3)
        has_zarr_v3 = (path / 'zarr.json').exists()
        has_zarr_v2 = (path / '.zarray').exists() or (path / '.zattrs').exists()
        
        if not (has_zarr_v3 or has_zarr_v2):
            raise ValueError(f"Directory does not appear to be a Zarr archive: {path}")
        
        logger.info(f"Reading Zarr archive: {path}")
        
        # Open with xarray
        ds_raw = xr.open_zarr(path)
        ds_raw = _promote_slope_metadata_coords(ds_raw)
        
        logger.info(
            f"  Loaded: {len(ds_raw.location):,} locations, "
            f"{len(ds_raw.time):,} timesteps, "
            f"{len(ds_raw.data_vars)} variables"
        )
        
        # Apply selectors if provided (already coerced by read())
        if time_sel:
            ds_raw = time_sel.apply(ds_raw)
        if location_sel:
            ds_raw = location_sel.apply(ds_raw)
        if slope_sel:
            ds_raw = slope_sel.apply(ds_raw)
        if realization_sel:
            ds_raw = realization_sel.apply(ds_raw)

        # Wrap in xsnowDataset
        return xsnowDataset(ds_raw)
        
    except (FileNotFoundError, ValueError, OSError) as e:
        # Expected failures: missing path, not a Zarr archive, or an I/O error
        # while reading. Per this function's contract, report and return None.
        logger.error(f"Failed to read Zarr archive: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
    except Exception:
        # Unexpected errors (e.g. a programming bug like a wrong xarray call)
        # must surface rather than be masked as a None "read failure".
        logger.exception(f"Unexpected error reading Zarr archive: {path}")
        raise


@register_xsnowDataset_method
def to_zarr(
    ds: xsnowDataset,
    path: Union[str, Path],
    mode: ZarrModes = 'w',
    consolidated: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs
):
    """
    Saves the dataset to a Zarr archive.

    Zarr is a directory-based format that provides several advantages over
    NetCDF for large datasets:
    - Better stability (pure Python, no HDF5 crashes)
    - Lower memory usage (streaming reads)
    - Cloud-friendly (works with S3, GCS out of the box)
    - Same xarray interface

    This method is particularly useful for very large datasets that may
    cause memory issues with NetCDF.

    Args:
        ds: The dataset instance to save.
        path: The destination path for the .zarr directory.
        mode: Write mode for Zarr. See xarray.Dataset.to_zarr() for details.
            'w' (default): Create new archive (overwrite if exists)
            'w-': Create new archive (fail if exists)
            'a': Append to existing archive
            'a-': Append (create if doesn't exist)
            'r+': Read/write existing archive
        consolidated: Whether to consolidate metadata for faster opening.
            Default is True, which is recommended for most use cases.
        logger: An optional preconfigured logger instance.
        **kwargs: Additional keyword arguments passed to `xarray.Dataset.to_zarr()`.

    Examples
    --------
    >>> ds = xsnow.read('large_data/')
    >>> ds.to_zarr('output.zarr')

    >>> # Append new data to existing archive
    >>> new_ds = xsnow.read('new_data/')
    >>> new_ds.to_zarr('output.zarr', mode='a')
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    
    logger = get_logger(__name__, logger=logger)
    
    # Ensure we're working with the underlying xarray Dataset
    ds_to_save = ds.to_xarray() if hasattr(ds, 'to_xarray') else ds.data
    
    # Ensure .zarr extension
    path = Path(path)
    if not str(path).endswith('.zarr'):
        path = Path(str(path) + '.zarr')
        logger.warning(f"Added .zarr extension: {path}")
    
    # Save to Zarr
    logger.info(f"Saving to Zarr: {path}")
    
    try:
        # Note: Zarr v3 uses default compression automatically
        # Custom encoding can be passed via kwargs if needed
        ds_to_save.to_zarr(
            path,
            mode=mode,
            consolidated=consolidated,
            **kwargs
        )
        
        logger.info(f"✅ Dataset saved to {path}")
        
        # Report size
        result = subprocess.run(
            ['du', '-sh', str(path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            logger.info(f"   Size: {size_str}")
        
    except Exception as e:
        logger.error(f"Failed to save Zarr archive: {e}")
        raise
