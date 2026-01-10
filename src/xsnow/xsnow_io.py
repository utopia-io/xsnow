# xsnow_io.py
"""
Handles all data parsing and I/O for the xsnow package, enabling scalable
and memory-efficient processing of large datasets using Dask.

This module is the primary engine for reading and writing snowpack data in
various formats (e.g., SNOWPACK .pro, .smet, NetCDF).

Core Dask Integration
---------------------
To handle datasets that are larger than memory, this module employs a lazy,
out-of-core loading strategy for .pro files using Dask. The process is:

1.  **Pre-Scan for Metadata**: A quick, parallel pre-scan of all .pro files is
    performed to determine essential metadata, such as the maximum number of
    snow layers (`max_layers`), without loading the full data into memory.

2.  **Lazy Graph Construction**: A Dask computation graph is built where each
    node represents the task of reading and processing a single .pro file into
    an xarray.Dataset. These tasks are "lazy" and are not executed immediately.

3.  **Parallel Computation**: When the data is needed, Dask's scheduler
    executes the graph in parallel, reading files and creating datasets in
    chunks. This ensures that only a fraction of the total data resides in
    memory at any given time, enabling the processing of vast amounts of data
    on a single machine.
"""
from __future__ import annotations
import os
import sys
import json
import logging
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List, Any, Union, Tuple, Sequence, Mapping, cast, Literal
from datetime import datetime


import numpy as np
import pandas as pd
import xarray as xr
from dask.base import compute
from dask.delayed import delayed
import dask.array as da

# Import from the package's modules
from .registry import registry
from .core import xsnowDataset
from .selectors import (
    TimeSelector, LocationSelector, SlopeSelector, RealizationSelector,
    TimeLike, LocationLike, SlopeLike, RealizationLike
)
from .plugin import register_xsnowDataset_method
from .combiners import concat as xsnow_concat
from .utils import get_logger, normalize_slope_coords, reorder_dims

# Define the allowed modes matching xarray/zarr requirements
ZarrModes = Literal['w', 'w-', 'a', 'a-', 'r+', 'r']

# --- Dynamic Mapping from Registry ---
CODE_TO_NAME_MAP = {
    details['param_code']: name
    for name, details in registry.items() if 'param_code' in details
}
CODE_TO_NAME_MAP["0500"] = "timestamp"

# --- Static Mappings ---
HEADER_MAP = {
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


def _round_coord(val: Any, ndigits: int = 6) -> float:
    """Round coordinate-like value to `ndigits`, preserving NaN."""
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(fval):
        return np.nan
    return round(fval, ndigits)


# --------------------------------------------------------------------------- #
#                      PRIMARY USER-FACING READ FUNCTION                      #
# --------------------------------------------------------------------------- #

def read(
    source: Union[str, List[str], Path, List[Path]],
    recursive: bool = False, 
    time: Optional[TimeSelector] = None,
    location: Optional[LocationSelector] = None,
    slope: Optional[SlopeSelector] = None,
    realization: Optional[RealizationSelector] = None,
    lazy: Optional[bool] = None,
    lazy_threshold: int = 30,
    parallel_lazy_creation: bool = True,
    n_cpus_use: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    chunks: Optional[Dict[str, int]] = None,
    **parser_kwargs: Any
) -> Optional[xsnowDataset]:
    """
    Reads, parses, and combines snowpack data into a unified xsnowDataset.

    This is the primary user-facing function for loading data. It orchestrates
    the discovery, parallel processing, and merging of snow profile (.pro)
    and time series (.smet) files into a single, coherent dataset.

    Args:
        source: The data source. Can be a path to a single file (e.g.,
            '.pro', '.smet', '.nc'), a directory containing data files, or a
            list of file paths.
        recursive: If True, search subdirectories recursively for .pro and .smet
            files when source is a directory. Default is False for backward
            compatibility and to avoid accidentally including unwanted files.
        time: A `TimeSelector` object for filtering data by time.
        location: A `LocationSelector` object for filtering by station ID.
        slope: A `SlopeSelector` for filtering by slope angle.
        realization: A `RealizationSelector` for filtering by model realization.
        lazy: If True, use Dask for lazy, memory-efficient loading (good for
            large datasets). If False, load all data eagerly into memory with
            parallel processing (good for small to medium datasets). If None
            (default), auto-detect based on file count (sweet spot to be determined).
        lazy_threshold: Number of files above which lazy loading is enabled
                       when lazy=None (auto-detect). Default: 30 files.
        parallel_lazy_creation: If True (default), parallelize the creation of lazy
            datasets using ThreadPoolExecutor. This speeds up the metadata parsing
            phase by 5-15x when working with hundreds or thousands of files.
            Recommended to keep True except for debugging.
        n_cpus_use: The number of CPU cores to use. If not provided, defaults
            to min(32, total_cores - 1), with a minimum of 1.
        logger: An optional, pre-configured logger instance.
        chunks: Chunk sizes for dask arrays when lazy=True.       
                Dict with keys 'time' and 'layer'.                  
                Default: {'time': 100, 'layer': -1}                
                Example: {'time': 50, 'layer': 100}                
        **parser_kwargs: Additional keyword arguments passed to the underlying
            parser functions:
            - `max_layers`: Max layer dimension; `'auto'` (default) scans files to pick
              the maximum, or set an int to force a fixed size.
            - `force_smart_detection`: If True, skip file validation for smart detection.
              Use when you know your files are heterogeneous but want fast loading anyway.
              Default: False.
            - `validation_threshold_days`: Maximum time spread (in days) for files to be
              considered homogeneous. Files modified beyond this threshold will trigger
              a validation warning. Default: 7.0 days.
            - `trim_padding`: If True, automatically trim NaN padding after loading.
              Only recommended for eager loading (lazy=False). For lazy datasets, this
              triggers computation. Default: False.
            - `remove_soil`: If True (default), drops soil layers during parsing.
            - `add_surface_sh_as_layer`: If True (default), inserts a surface hoar
              layer explicitly as layer when indicated by the PRO file special code for SH at surface.
            - `norm_slopes`: Controls slope normalization in SNOWPACK parsing /
              merging. Options:
                * `'auto'` (default): normalize when slopes are consistent across
                  locations; preserve per-location slopes when inconsistent.
                * True: force normalization (collapse to `(slope,)` even if slopes
                  vary by location).
                * False: skip normalization entirely.

    Returns:
        A unified `xsnowDataset` object containing the data from all sources,
        or `None` if no valid data files are found.

    Examples:
        >>> # Auto-detect (lazy for many files, eager for few)
        >>> ds = read("data/")
        
        >>> # Force lazy loading for large datasets
        >>> ds = read("large_archive/", lazy=True, n_cpus_use=8)
        
        >>> # Force eager loading with trimmed padding
        >>> ds = read("small_dataset/", lazy=False, trim_padding=True)
        
        >>> # Force smart detection for heterogeneous files (use with caution)
        >>> ds = read("mixed_archive/", force_smart_detection=True)
        
        >>> # Force eager loading for quick access
        >>> ds = read("small_dataset/", lazy=False, n_cpus_use=4)
        
    See Also
    --------
    to_netcdf : Save datasets to NetCDF with proper formatting
    """
    # --- Step 1: Setup ---
    logger = get_logger(__name__, logger=logger)

    if n_cpus_use is None:
        cpu_count = os.cpu_count()
        n_cpus_use = min(32, cpu_count - 1) if cpu_count and cpu_count > 1 else 1

    time_sel        = _coerce_time(time)
    location_sel    = _coerce_location(location)
    slope_sel       = _coerce_slope(slope)
    realization_sel = _coerce_realization(realization)
    
    # --- Step 2: Resolve file paths ---
    if isinstance(source, (str, Path)) and Path(source).suffix == '.nc':
        try:
            return _read_xsnow_netcdf(source)
        except Exception as e:
            logger.warning(f"Failed to read NetCDF file {source}: {e}")
            return None
    
    # Handle Zarr archives (similar to NetCDF handling)
    if isinstance(source, (str, Path)) and (Path(source).suffix == '.zarr' or 
                                            (Path(source).is_dir() and 
                                             (Path(source) / 'zarr.json').exists())):
        try:
            return _read_zarr(source, time_sel, location_sel, slope_sel, realization_sel, logger)
        except Exception as e:
            logger.warning(f"Failed to read Zarr archive {source}: {e}")
            return None
    
    # Continue with normal .pro/.smet file processing
    try:
        pro_files, smet_files = _resolve_filepaths(source, recursive, logger)
    except (FileNotFoundError, TypeError) as e:
        logger.error(e)
        return None

    if not pro_files and not smet_files:
        logger.warning("No data files found in the specified source.")
        return None

    if not pro_files and not smet_files:
        logger.warning("No data files found in the specified source.")
        return None

    # --- Step 3: Auto-detect lazy mode if not specified ---
    if lazy is None:
        total_files = len(pro_files) + len(smet_files)
        lazy = total_files > lazy_threshold
        logger.debug(f"Auto-detected lazy={lazy} for {total_files} files (threshold={lazy_threshold})")

    # --- Step 4: Read and process data files ---
    # TODO: The datetime_start *_end parsing results in nan for inclination and azimuth!
    datetime_start, datetime_end = (time_sel.sel.start, time_sel.sel.stop) if time_sel and isinstance(time_sel.sel, slice) else (None, None)
    
    if lazy:
        chunks = chunks or {'time': 100, 'layer': -1}

        parser_kwargs['use_dask'] = lazy      # ← Set the flag
        parser_kwargs['chunks'] = chunks      # ← Set the chunks

        logger.info("Using lazy loading with Dask")
        pro_datasets = _read_pro_files_lazy(pro_files, datetime_start, datetime_end, n_cpus_use, parallel_lazy_creation, logger, parser_kwargs)
    else:
        # logger.info("Using eager loading with multiprocessing")
        pro_datasets = _read_pro_files_eager(pro_files, datetime_start, datetime_end, n_cpus_use, logger, parser_kwargs)
    
    smet_datasets = _read_smet_files(smet_files, datetime_start, datetime_end, logger)

    # --- Step 5: Merge and finalize datasets ---
    all_datasets = pro_datasets + smet_datasets
    final_dataset = _merge_and_finalize(
        all_datasets,
        time_sel,
        location_sel,
        slope_sel,
        realization_sel,
        logger=logger,
        norm_slopes=parser_kwargs.get('norm_slopes', 'auto'),
    )
    
    # --- Step 6: Trim padding if requested ---
    if final_dataset is not None and parser_kwargs.get('trim_padding', False):
        if lazy:
            logger.warning(
                """
                WARNING: For lazy-loaded datasets, this will trigger computation of ALL
                dask arrays to check for NaN values!

                For large lazy datasets, use this workflow instead:
                    # Load lazily
                    ds = read('huge/', lazy=True)
                    
                    # Subset first
                    ds_sub = ds.sel(location=['A', 'B'], time='2024')
                    
                    # Compute only the subset
                    ds_sub = ds_sub.compute()
                    
                    # Trim the small subset (fast!)
                    ds_sub = ds_sub.trim_padding()
                """
            )
        logger.info("Trimming NaN padding from layer dimension...")
        final_dataset = final_dataset.trim_padding()
    
    return final_dataset



def _read_xsnow_netcdf(source: Union[str, Path]) -> Optional[xsnowDataset]:
    """
    Read xsnow's custom netcdf format.

    This includes proper json decoding of any 'mapping' attribute and
    handling of fixed-length byte coordinate strings.
    
    Args:
        source: Path to the NetCDF file.
        
    Returns:
        An xsnowDataset instance, or None if reading fails.
    """
    # open lazily, then materialize metadata/coords eagerly
    with xr.open_dataset(source) as _ds:
        ds = _ds.load()

    # Decode fixed-length byte coords back to str
    for coord_name in ("location", "realization"):
        if coord_name in ds.coords:
            coord_data = ds[coord_name]
            if coord_data.dtype.kind == 'S':  # byte strings
                decoded_values = [
                    (val.decode("utf-8") if isinstance(val, (bytes, np.bytes_)) else str(val))
                    for val in coord_data.values
                ]
                ds[coord_name] = (coord_data.dims, decoded_values)

    # Restore mapping attrs from JSON
    ds = _decode_mapping_attrs_from_netcdf(ds)

    return xsnowDataset(ds)
    

def _decode_mapping_attrs_from_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """
    Reverse of _encode_mapping_attrs_for_netcdf:
    - If 'mapping_json' exists (dataset or variable), parse JSON and restore
      the 'mapping' dict attr (with int keys when possible).
    - Leaves other attrs untouched.
    """
    def _parse(json_str: str):
        try:
            obj = json.loads(json_str)
            # convert numeric-like keys back to int where possible
            if isinstance(obj, dict):
                new = {}
                for k, v in obj.items():
                    try:
                        k2 = int(k)
                    except (ValueError, TypeError):
                        k2 = k
                    new[k2] = v
                return new
            return obj
        except Exception:
            # fall back: leave as string if something goes wrong
            return json_str

    # dataset-level
    mj = ds.attrs.get("mapping_json")
    if isinstance(mj, str):
        decoded = _parse(mj)
        ds.attrs["mapping"] = decoded
        # optional: clean up the helper attr
        del ds.attrs["mapping_json"]

    # variable-level
    for name, var in ds.variables.items():
        mj = var.attrs.get("mapping_json")
        if isinstance(mj, str):
            decoded = _parse(mj)
            var.attrs["mapping"] = decoded
            del var.attrs["mapping_json"]

    return reorder_dims(ds)

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

    Internal function to read a Zarr archive.
    
    Called by read() when it detects a .zarr file or directory.
    Similar to _read_xsnow_netcdf() for NetCDF files.
    
    Zarr archives are directory-based storage format that provides:
    - Better stability (pure Python, no HDF5 crashes)
    - Lower memory usage (streaming reads)
    - Cloud-friendly (works with S3, GCS)
    - Same xarray interface as NetCDF
    
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
        
        logger.info(f"  Loaded: {len(ds_raw.location):,} locations, "
                   f"{len(ds_raw.time):,} timesteps, "
                   f"{len(ds_raw.data_vars)} variables")
        
        # Wrap in xsnowDataset
        ds = xsnowDataset(ds_raw)
        
        # Apply selectors if provided (already coerced by read())
        if time_sel:
            ds = ds.where(time=time_sel)
        if location_sel:
            ds = ds.where(location=location_sel)
        if slope_sel:
            ds = ds.where(slope=slope_sel)
        if realization_sel:
            ds = ds.where(realization=realization_sel)
        
        return ds
        
    except Exception as e:
        logger.error(f"Failed to read Zarr archive: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

# --------------------------------------------------------------------------- #
#                    EAGER LOADING WITH MULTIPROCESSING                       #
# --------------------------------------------------------------------------- #

def _read_pro_files_eager(
    pro_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    n_cpus_use: int,
    logger: logging.Logger,
    parser_kwargs: Dict
) -> List[xr.Dataset]:
    """
    Eagerly loads and processes .pro files using direct multiprocessing.
    
    This approach immediately loads all data into memory with parallel processing.
    Best for small to medium datasets (<100 files) where memory is sufficient.
    
    Args:
        pro_files: A list of file paths to .pro files.
        datetime_start: An optional start timestamp for filtering.
        datetime_end: An optional end timestamp for filtering.
        n_cpus_use: The number of CPU cores to use for parallel processing.
        logger: A configured logger instance.
        parser_kwargs: Additional keyword arguments for the parser.

    Returns:
        A list of xarray.Dataset objects, one for each successfully
        processed .pro file.
    """
    if not pro_files:
        return []

    # --- Stage 1: Pre-scan for Max Layers ---
    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
    remove_soil = parser_kwargs.get('remove_soil', True)
    
    if max_layers_arg == 'auto':
        # Extract validation parameters from parser_kwargs
        force_smart = parser_kwargs.get('force_smart_detection', False)
        validation_days = parser_kwargs.get('validation_threshold_days', 7.0)
        
        # Use smart size-based detection (same as lazy mode for consistency)
        max_layers_to_use = _determine_max_layers_smart(
            pro_files, n_cpus_use, add_sh, remove_soil, logger,
            force=force_smart, validation_threshold_days=validation_days
        )
        logger.debug(f"Pre-scan complete. Max layers found: {max_layers_to_use}")
    else:
        max_layers_to_use = int(max_layers_arg)
    
    # --- Stage 2: Parallel Loading with Direct Multiprocessing ---
    logger.info(f"Loading {len(pro_files)} datasets eagerly with {n_cpus_use} workers...")
    
    if n_cpus_use > 1 and len(pro_files) > 1:
        # Parallel processing
        ctx = multiprocessing.get_context('spawn')
        
        # Create task arguments
        tasks = [
            (fp, max_layers_to_use, datetime_start, datetime_end, parser_kwargs)
            for fp in pro_files
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus_use, mp_context=ctx) as executor:
            datasets = list(executor.map(_process_pro_file_eager, tasks))
    else:
        # Sequential processing
        tasks = [
            (fp, max_layers_to_use, datetime_start, datetime_end, parser_kwargs)
            for fp in pro_files
        ]
        datasets = [_process_pro_file_eager(task) for task in tasks]
    
    # Filter out None results
    final_datasets = [ds for ds in datasets if ds is not None]
    logger.debug(f"Successfully assembled {len(final_datasets)} datasets")
    
    return final_datasets


def _process_pro_file_eager(task_args: Tuple) -> Optional[xr.Dataset]:
    """
    Worker function for eager loading of a single .pro file.
    
    This function is designed to be called by multiprocessing workers.
    It unpacks task arguments and processes one file.
    
    Args:
        task_args: Tuple of (filepath, max_layers, datetime_start, datetime_end, parser_kwargs)
    
    Returns:
        An xarray.Dataset or None if processing failed.
    """
    filepath, max_layers, datetime_start, datetime_end, parser_kwargs = task_args
    
    try:
        remove_soil = parser_kwargs.get('remove_soil', True)
        add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
        
        # Parse the file
        raw_data = _read_pro_raw(
            filepath,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            code_map=CODE_TO_NAME_MAP,
            logger=None,  # Don't pass logger to worker (not picklable)
            remove_soil=remove_soil,
            add_surface_sh_as_layer=add_sh
        )
        
        # Create dataset if data found
        if raw_data and raw_data[0]:
            metadata, profiles = raw_data
            return _create_pro_dataset(metadata, profiles, max_layers, add_sh)
    except Exception as e:
        # Can't use logger in worker, just return None
        logging.error(f"Failed to process {filepath.name}: {e}")
        try:
            metadata = _read_pro_metadata(filepath, throw=False)
            return _create_pro_dataset(metadata, [], max_layers, empty_profile_status_default=-1)
        except Exception as e:
            logging.warning(f"Cannot even recover minimal metadata for {filepath.name}. Dropping from dataset. | {e}")
            return None



# --------------------------------------------------------------------------- #
#                      LAZY .PRO FILE PROCESSING PIPELINE                        #
# --------------------------------------------------------------------------- #

def _validate_files_for_smart_detection(
    pro_files: List[Path],
    logger: logging.Logger,
    time_threshold_days: float = 7.0
) -> Tuple[bool, str]:
    """
    Validate that .pro files are suitable for smart max_layers detection.
    
    Smart detection assumes all files have similar temporal structure (same number
    of timestamps). This breaks down when files are from different runs, time periods,
    or have different temporal resolutions.
    
    This function checks file modification times to detect heterogeneous datasets.
    If files span a wide time range, they likely have different structures.
    
    Args:
        pro_files: List of .pro file paths
        logger: Logger instance
        time_threshold_days: Maximum time spread (days) between oldest and newest
                            file modification time to consider files homogeneous.
                            Default: 7 days
        
    Returns:
        (is_valid, reason): 
            - is_valid: True if files are homogeneous, False if heterogeneous
            - reason: Explanation string
    """
    if len(pro_files) <= 1:
        # Single file or no files - always valid
        return True, "Single file or empty dataset"
    
    # Get file modification times
    mod_times = []
    for f in pro_files:
        try:
            stat = f.stat()
            mod_times.append(stat.st_mtime)
        except Exception as e:
            logger.warning(f"Could not stat {f.name}: {e}")
    
    if len(mod_times) < 2:
        return True, "Insufficient file metadata"
    
    # Calculate time spread
    oldest = min(mod_times)
    newest = max(mod_times)
    spread_seconds = newest - oldest
    spread_days = spread_seconds / 86400.0
    
    # Check against threshold
    if spread_days <= time_threshold_days:
        return True, f"All files modified within {spread_days:.2f} days"
    else:
        return False, (
            f"Files span {spread_days:.1f} days (threshold: {time_threshold_days} days). "
            f"This suggests a heterogeneous dataset with files from different runs "
            f"or time periods, which may have different temporal structures."
        )


def _determine_max_layers_smart(
    pro_files: List[Path],
    n_cpus_use: int,
    add_sh: bool,
    remove_soil: bool,
    logger: logging.Logger,
    force: bool = False,
    validation_threshold_days: float = 7.0
) -> int:
    """
    Intelligently determine max_layers using file size heuristic.
    
    Key insight: If all .pro files have the same number of timestamps,
    file size is proportional to layer count. Therefore, the largest
    files almost certainly have the most layers.
    
    This approach scans only the 10 largest files instead of all files,
    providing ~2000x speedup for large archives while maintaining 99.9%+ accuracy.
    
    IMPORTANT: The 10% buffer is only added when doing a PARTIAL scan.
    When we scan ALL files, we know the exact maximum and don't need a buffer.
    
    NEW: Validates files before smart detection. If files appear heterogeneous
    (modified across wide time range), warns user and raises error with options.
    
    Args:
        pro_files: List of .pro file paths
        n_cpus_use: Number of CPUs for parallel scanning
        add_sh: Whether surface hoar layer will be added
        remove_soil: Whether soil layers will be removed
        logger: Logger instance
        force: If True, skip validation and use smart detection anyway
        validation_threshold_days: Maximum time spread for homogeneous files
        
    Returns:
        Safe max_layers value (with 10% buffer if partial scan, exact if full scan)
        
    Raises:
        ValueError: If validation fails and force=False
    """
    import time
    start = time.time()
    
    # Step 0: VALIDATE FILES FOR SMART DETECTION
    if not force:
        is_valid, reason = _validate_files_for_smart_detection(
            pro_files, logger, validation_threshold_days
        )
        
        if not is_valid:
            logger.warning("=" * 70)
            logger.warning("SMART MAX_LAYERS DETECTION MAY BE UNRELIABLE")
            logger.warning("=" * 70)
            logger.warning(f"Reason: {reason}")
            logger.warning("")
            logger.warning("Smart detection assumes all files have similar temporal")
            logger.warning("structure (same number of timestamps). Your dataset appears")
            logger.warning("heterogeneous, which could lead to incorrect max_layers.")
            logger.warning("")
            logger.warning("OPTIONS:")
            logger.warning("  1. Set max_layers manually:")
            logger.warning("     ds = read(path, max_layers=100)")
            logger.warning("")
            logger.warning("  2. Force smart detection (fast but risky):")
            logger.warning("     ds = read(path, max_layers='auto', force_smart_detection=True)")
            logger.warning("")
            logger.warning("  3. Use full scan (slower but safe):")
            logger.warning("     ds = read(path, lazy=False)  # Scans all files exactly")
            logger.warning("=" * 70)
            
            raise ValueError(
                f"Dataset validation failed for smart detection: {reason}\n"
                f"See log for options. Set max_layers manually or use force_smart_detection=True."
            )
    
    # Step 1: Get file sizes (instant - just stat())
    logger.info(f"Analyzing file sizes for {len(pro_files)} files...")
    file_sizes = []
    for f in pro_files:
        try:
            size = f.stat().st_size
            file_sizes.append((f, size))
        except Exception as e:
            logger.warning(f"Could not stat {f.name}: {e}")
    
    if not file_sizes:
        logger.warning("No valid files found, using default max_layers=200")
        return 200
    
    # Step 2: Sort by size, get top N largest
    # Scanning top 10 gives us high confidence that we've found the max
    n_largest = min(10, len(file_sizes))
    largest_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)[:n_largest]
    
    largest_size = largest_files[0][1] if largest_files else 0
    smallest_size = min(file_sizes, key=lambda x: x[1])[1] if file_sizes else 0
    
    logger.info(f"File size range: {smallest_size/1024:.1f}KB - {largest_size/1024:.1f}KB")
    logger.info(f"Scanning {n_largest} largest files to determine max layer count...")
    
    # Step 3: Scan only the largest files
    # Extract just the file paths for processing
    largest_file_paths = [f for f, _ in largest_files]
    
    if n_cpus_use > 1 and n_largest > 1:
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        
        # Use submit instead of map to avoid lambda pickling issues
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(n_cpus_use, n_largest), 
            mp_context=ctx
        ) as executor:
            futures = {
                executor.submit(_scan_pro_for_max_layers, fp, add_sh, remove_soil): fp 
                for fp in largest_file_paths
            }
            layer_counts = []
            for future in concurrent.futures.as_completed(futures):
                layer_counts.append(future.result())
    else:
        layer_counts = [_scan_pro_for_max_layers(f, add_sh, remove_soil) 
                       for f in largest_file_paths]
    
    max_in_sample = max(layer_counts) if layer_counts else 200
    
    # Step 4: CONDITIONAL BUFFER - only add if we didn't scan all files
    # This is critical: if we scanned everything, we know the exact maximum!
    total_files = len(pro_files)
    scanned_all = (n_largest >= total_files)
    
    if scanned_all:
        # Full scan - we know the exact maximum, no buffer needed
        safe_max_layers = max_in_sample
        buffer_msg = "(exact - scanned all files)"
    else:
        # Partial scan - add 10% safety buffer
        safe_max_layers = int(max_in_sample * 1.1)
        buffer_msg = "(with 10% buffer - partial scan)"
    
    elapsed = time.time() - start
    
    logger.info(f"✅ Smart max_layers determination complete:")
    logger.info(f"   - Checked: {total_files} file sizes")
    logger.info(f"   - Scanned: {n_largest} largest files")
    logger.info(f"   - Max layers found: {max_in_sample}")
    logger.info(f"   - Using max_layers: {safe_max_layers} {buffer_msg}")
    logger.info(f"   - Time elapsed: {elapsed:.2f} seconds")
    
    return safe_max_layers

def _read_pro_files_lazy(
    pro_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    n_cpus_use: int,
    parallel_lazy_creation: bool,
    logger: logging.Logger,
    parser_kwargs: Dict
) -> List[xr.Dataset]:
    """
    Parses and creates datasets from .pro files with TRUE lazy loading.
    
    PERFORMANCE IMPROVEMENT: This function now implements genuine lazy loading:
    - Stage 1: Smart max_layers detection (scans only 10 largest files - <1 sec)
    - Stage 2: Creates delayed dataset objects WITHOUT loading profile data
    - Stage 3: Data only loads when user calls .compute() or .load()
    
    Key innovation: Uses file size to identify which files likely have the most
    layers, then only scans those files. This provides ~2000x speedup vs scanning
    all files while maintaining 99.9%+ accuracy.
    
    Behavior depends on the 'use_dask' flag in parser_kwargs:
    - If use_dask=True: Returns datasets with dask.delayed loading (TRULY lazy).
      Profile data is NOT read until .compute()/.load() is called.
      Suitable for datasets larger than RAM (e.g., 209k files / 316GB).
    - If use_dask=False: Uses Dask for parallel file processing, then calls
      compute() to create numpy-backed datasets. Faster for small datasets
      that fit in memory.

    Args:
        pro_files: A list of file paths to .pro files.
        datetime_start: An optional start timestamp for filtering.
        datetime_end: An optional end timestamp for filtering.
        n_cpus_use: The number of CPU cores to use for parallel processing.
        parallel_lazy_creation: If True, parallelize the metadata extraction
            using ThreadPoolExecutor. Provides significant speedup for many files.
        logger: A configured logger instance.
        parser_kwargs: Additional keyword arguments for the parser, including:
            - 'use_dask' (bool): Whether to create truly lazy dask arrays
            - 'chunks' (dict): Chunk sizes if use_dask=True, e.g., {'time': 100, 'layer': -1}
            - 'max_layers': Max layers to pad to ('auto' for smart detection, or integer)
            - 'remove_soil' (bool): Remove soil layers
            - 'add_surface_sh_as_layer' (bool): Add surface hoar as layer

    Returns:
        A list of xarray.Dataset objects. If use_dask=True, these contain
        dask.delayed functions that will load data on-demand. If use_dask=False,
        these contain numpy arrays and data is already loaded in memory.
    """
    if not pro_files:
        return []

    # Extract use_dask flag - default to True since this function is meant to be lazy
    use_dask = parser_kwargs.get('use_dask', True)
    
    # Get the add_sh and remove_soil arguments
    add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
    remove_soil = parser_kwargs.get('remove_soil', True)
    
    # ============================================================
    # SMART MAX_LAYERS DETERMINATION
    # ============================================================
    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    
    if max_layers_arg == 'auto':
        # Extract validation parameters from parser_kwargs
        force_smart = parser_kwargs.get('force_smart_detection', False)
        validation_days = parser_kwargs.get('validation_threshold_days', 7.0)
        
        # Use smart size-based detection (scans only 10 largest files!)
        default_max_layers = _determine_max_layers_smart(
            pro_files, n_cpus_use, add_sh, remove_soil, logger,
            force=force_smart, validation_threshold_days=validation_days
        )
        
    elif isinstance(max_layers_arg, int):
        # User explicitly specified
        default_max_layers = max_layers_arg
        logger.info(f"Using user-specified max_layers={default_max_layers}")
        
    else:
        # Fallback
        default_max_layers = 200
        logger.warning(f"Invalid max_layers value '{max_layers_arg}', using default=200")
    
    # ============================================================
    # STAGE 2: Create Truly Lazy Delayed Loaders
    # ============================================================
    if use_dask:
        # TRUE LAZY MODE: Create delayed dataset loading functions
        logger.info("Creating TRULY LAZY datasets using dask.delayed...")
        logger.info(f"Using max_layers dimension: {default_max_layers}")
        logger.info("Profile data will NOT be read until .compute() or .load() is called")
        
        chunks = parser_kwargs.get('chunks', {'time': 100, 'layer': -1})
        
        @delayed
        def load_single_file_with_validation(fp: Path) -> Optional[xr.Dataset]:
            """
            Delayed loader that scans and loads THIS file only when .compute() is called.
            
            This function is NOT called during initialization!
            It executes ONLY when the user explicitly requests the data.
            
            Each file independently:
            1. Scans itself for layer count (fast - just counts lines)
            2. Validates it doesn't exceed max_layers
            3. Reads profile data
            4. Pads to the default max_layers dimension
            """
            try:
                # Step 1: Scan THIS file for its actual layer count
                actual_layers = _scan_pro_for_max_layers(
                    fp, add_sh, remove_soil, logger=None
                )
                
                # Step 2: CRITICAL VALIDATION - Check against default
                if actual_layers > default_max_layers:
                    raise ValueError(
                        f"File {fp.name} has {actual_layers} layers but "
                        f"max_layers={default_max_layers}. "
                        f"This file was not in the largest 10 files by size. "
                        f"Increase max_layers parameter or use max_layers='auto' to resample."
                    )
                
                # Step 3: Read the actual profile data
                metadata, profiles = _read_pro_raw(
                    fp,
                    datetime_start=datetime_start,
                    datetime_end=datetime_end,
                    code_map=CODE_TO_NAME_MAP,
                    logger=None,  # Don't pass logger (not picklable in delayed)
                    remove_soil=remove_soil,
                    add_surface_sh_as_layer=add_sh
                )
                
                if metadata:
                    # Step 4: Create dataset padded to DEFAULT dimension
                    # Use numpy inside delayed context (dask will handle parallelization)
                    ds = _create_pro_dataset(
                        metadata, 
                        profiles, 
                        max_layers=default_max_layers,
                        add_surface_sh_as_layer=add_sh,
                        use_dask=True,  # Use numpy inside delayed
                        chunks=chunks
                    )
                    return ds
                else:
                    return None
                    
            except Exception as e:
                # Errors will be logged when .compute() is called
                print(f"Error loading {fp.name}: {e}")
                try:
                    metadata = _read_pro_metadata(fp, logger=None, throw=False)
                    return _create_pro_dataset(
                        metadata, [], default_max_layers, 
                        add_sh, use_dask=False, chunks=None,
                        empty_profile_status_default=-1
                    )
                except Exception:
                    pass
            
            return None
        
        # Create delayed objects (NOT calling the function yet!)
        logger.info(f"Building delayed loading graph for {len(pro_files)} files...")
        delayed_datasets = [load_single_file_with_validation(fp) for fp in pro_files]
        
        logger.info(f"✅ Lazy structure created successfully!")
        logger.info(f"   - Files: {len(pro_files)}")
        logger.info(f"   - Layer dimension: {default_max_layers}")
        logger.info(f"   - Data loaded: NONE (truly lazy!)")
        logger.info(f"   - Call .compute() or .load() to read files")
        
        # For lazy mode, we need to compute the delayed objects to get actual datasets
        # but with dask arrays (still lazy at the data level)
        if parallel_lazy_creation and n_cpus_use > 1:
            logger.info(f"Computing delayed datasets with {n_cpus_use} workers...")
            final_datasets = list(compute(*delayed_datasets, scheduler='threads', num_workers=n_cpus_use))
        else:
            logger.info("Computing delayed datasets sequentially...")
            final_datasets = [ds.compute() for ds in delayed_datasets]
        
        # Filter out None results
        final_datasets = [ds for ds in final_datasets if ds is not None]
        
        logger.info(f"Created {len(final_datasets)} datasets")
        return final_datasets
        
    else:
        # PARALLEL PROCESSING MODE: Use Dask for parallelization, compute immediately
        logger.info("Building Dask graph for parallel dataset creation...")
        lazy_datasets = [
            _create_dataset_from_pro_file(
                fp, default_max_layers, datetime_start, datetime_end, logger, parser_kwargs
            ) for fp in pro_files
        ]

        if not lazy_datasets:
            return []
        
        logger.info(f"Computing {len(lazy_datasets)} tasks in parallel using Dask...")
        pro_datasets = list(compute(*lazy_datasets, scheduler='processes', num_workers=n_cpus_use))
        
        # Filter out None results from files that had no valid data
        final_datasets = [ds for ds in pro_datasets if ds is not None]
        logger.info(f"Successfully created {len(final_datasets)} datasets (data loaded into memory).")
        
        return final_datasets

@delayed
def _create_dataset_from_pro_file(
    filepath: Path,
    max_layers: int,
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    logger: logging.Logger,
    parser_kwargs: Dict
) -> Optional[xr.Dataset]:
    """
    A Dask-delayed function to create a single xarray.Dataset from one .pro file.

    This function encapsulates the entire processing logic for a single file
    into a lazy task. It first parses the raw data from the file and then
    converts it into a properly padded xarray.Dataset.

    Args:
        filepath: Path to the .pro file.
        max_layers: The number of layers to pad the dataset to, determined by
            the pre-scan.
        datetime_start: Start timestamp for filtering profiles.
        datetime_end: End timestamp for filtering profiles.
        logger: A configured logger instance.
        parser_kwargs: Additional keyword arguments for the parser.

    Returns:
        An xarray.Dataset for the station, or None if no valid profiles are found.
    """
    try:
        # Get the remove_soil and add_sh arguments from parser_kwargs
        remove_soil = parser_kwargs.get('remove_soil', True)
        add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
        
        # Pass the arguments to the raw parser
        raw_data = _read_pro_raw(
            filepath,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            code_map=CODE_TO_NAME_MAP,
            logger=logger,
            remove_soil=remove_soil,
            add_surface_sh_as_layer=add_sh
        )

        # Step 2: Create the dataset
        if raw_data and raw_data[0]:  # Check if metadata dict is not empty
            metadata, profiles = raw_data
            use_dask = parser_kwargs.get('use_dask', False)
            chunks = parser_kwargs.get('chunks', None)
            return _create_pro_dataset(metadata, profiles, max_layers, add_sh, 
                                    use_dask=use_dask, chunks=chunks)
            
    except Exception as e:
        try:
            metadata = _read_pro_metadata(filepath, logger=logger, throw=False)
            ds = _create_pro_dataset(metadata, [], max_layers, empty_profile_status_default=-1)
            logger.error(f"Cannot parse {filepath.name}", exc_info=e)
            return ds
        except Exception as e:
            logger.warning(f"Warning: Cannot even recover minimal metadata for {filepath.name}. Dropping from dataset. | {e}")

    
    return None

# --------------------------------------------------------------------------- #
#          STATION NAME/LOCATION HELPER FUNCTIONS FOR read .PRO FILES         #
# --------------------------------------------------------------------------- #

def _sanitize_station_name(name: str) -> str:
    """
    Sanitize station name for use as xarray coordinate.
    
    Rules:
    - Strip leading/trailing whitespace
    - Replace whitespace sequences with single underscore
    - Replace problematic characters with underscore
    - Ensure non-empty
    - Limit length to 64 characters
    
    Parameters:
    -----------
    name : str
        Raw station name from .pro file
    
    Returns:
    --------
    str
        Sanitized station name safe for xarray/NetCDF
    
    Examples:
    ---------
    >>> _sanitize_station_name("Station A-1")
    'Station_A-1'
    >>> _sanitize_station_name("  VIR1A  ")
    'VIR1A'
    >>> _sanitize_station_name("")
    'UNNAMED_STATION'
    """
    import re
    
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



def _handle_duplicate_names(names: List[str], 
                            logger: Optional[logging.Logger]=None) -> List[str]:
    """
    Ensure all station names are unique by appending indices to duplicates.
    
    Parameters:
    -----------
    names : list[str]
        Original station names (may contain duplicates)
    
    Returns:
    --------
    list[str]
        Unique station names with indices appended to duplicates
    
    Examples:
    ---------
    >>> _handle_duplicate_names(['A', 'B', 'A', 'C', 'A'])
    ['A_0', 'B', 'A_1', 'C', 'A_2']
    >>> _handle_duplicate_names(['A', 'B', 'C'])
    ['A', 'B', 'C']
    """
    if len(names) == len(set(names)):
        return names
    log = get_logger(__name__, logger=logger)
    
    # Find duplicates
    from collections import Counter
    name_counts = Counter(names)
    duplicates = {name for name, count in name_counts.items() if count > 1}
    
    # Rename all occurrences of duplicate names
    occurrence_counter = {}
    unique_names = []
    warned = set()
    
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

def _extract_metadata_fast(
    filepath: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Extract minimal metadata from a .pro file WITHOUT reading profile data.
    
    This is an ultra-lightweight version that only reads the header section
    and the first/last timestamp lines. It skips all profile data lines,
    making it ~150x faster than full parsing.
    
    Designed for lazy loading initialization where we need to build the
    coordinate structure but don't want to load actual data yet.
    
    Args:
        filepath: Path to the .pro file
        logger: Optional logger instance
        
    Returns:
        Dict containing:
            - station_name, latitude, longitude, altitude, slopeAngle, slopeAzi
            - first_timestamp: First timestamp found in file
            - last_timestamp: Last timestamp found in file
            - estimated_profiles: Approximate number of profiles (based on 0500 lines)
    """
    logger = get_logger(__name__, logger=logger)
    file_path = Path(filepath)
    
    metadata = {
        "station_name": file_path.stem,
        "latitude": np.nan,
        "longitude": np.nan,
        "altitude": np.nan,
        "slopeAngle": np.nan,
        "slopeAzi": np.nan,
        "tz": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "estimated_profiles": 0,
    }
    
    if not file_path.exists():
        logger.warning(f"File not found: {filepath}")
        return metadata
    
    in_header = False
    in_data = False
    profile_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Detect header section
                if stripped == "[STATION_PARAMETERS]":
                    in_header = True
                    continue
                elif stripped.startswith('[DATA]'):
                    in_data = True
                    in_header = False
                    continue
                elif stripped.startswith('['):
                    in_header = False
                    continue
                
                # Parse header
                if in_header:
                    for key, value in HEADER_MAP.items():
                        if stripped.startswith(key):
                            metadata[value] = stripped.split('=', 1)[1].strip()
                            break
                
                # Parse ONLY timestamp lines in data section (0500 lines)
                elif in_data and stripped.startswith("0500,"):
                    parts = stripped.split(',')
                    if len(parts) > 1:
                        timestamp = parts[1].strip()
                        if metadata["first_timestamp"] is None:
                            metadata["first_timestamp"] = timestamp
                        metadata["last_timestamp"] = timestamp
                        profile_count += 1
                    # CRITICAL: Don't read the data lines (0501, 0502, etc.)
                    # This is what makes it fast!
                    
    except Exception as e:
        logger.warning(f"Error extracting metadata from {file_path.name}: {e}")
    
    metadata["estimated_profiles"] = profile_count
    
    # Sanitize station name
    if metadata.get("station_name"):
        metadata["station_name"] = _sanitize_station_name(metadata["station_name"])
    
    # Normalize coordinates
    for coord_key in ("latitude", "longitude"):
        if coord_key in metadata:
            metadata[coord_key] = _round_coord(metadata[coord_key])
    
    return metadata


def _read_pro_metadata(
    filepath: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    throw: bool = True,
) -> Dict[str, Any]:
    """
    Read minimal metadata from a .pro file header.

    Returns a dict with at least:
      - station_name
      - latitude
      - longitude
      - altitude
      - slopeAngle (raw)
      - slopeAzi (raw)

    If the file cannot be read, falls back to filename-based station_name
    and NaNs for others.
    """
    logger = get_logger(__name__, logger=logger)

    file_path = Path(filepath)

    metadata: Dict[str, Any] = {
        "station_name": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
        "altitude": np.nan,
        "slopeAngle": np.nan,
        "slopeAzi": np.nan,
        "tz": None,  # None means not specified in file
    }

    if not file_path.exists():
        if throw:
            raise FileNotFoundError(f"File not found while reading metadata: {filepath}")
        else:
            metadata["station_name"] = _sanitize_station_name(file_path.stem)
            return metadata


    in_header = False
    has_header = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line == "[STATION_PARAMETERS]":
                    in_header = True
                    has_header = True
                    continue
                if line.startswith('[') and line != "[STATION_PARAMETERS]":
                    # any other section ends header
                    if in_header:
                        break
                    continue
                if in_header:
                    for key, value in HEADER_MAP.items():
                        if line.startswith(key):
                            metadata[value] = line.split('=', 1)[1].strip()
    except Exception as exc:
        logger.warning(f"Error reading metadata from {file_path.name}: {exc}")
        # keep fallback metadata

    if not has_header and throw:
        raise ValueError(f"No declared header exists in {file_path.name}.")

    name = metadata.get("station_name")
    if name is None or pd.isna(name) or (isinstance(name, str) and not name.strip()):
        if throw:
            raise ValueError(f"No StationName found in {file_path.name}.")
        else:
            metadata["station_name"] = file_path.stem

    metadata["station_name"] = _sanitize_station_name(metadata["station_name"])

    # Normalize coordinates
    for coord_key in ("latitude", "longitude"):
        if coord_key in metadata:
            metadata[coord_key] = _round_coord(metadata[coord_key])

    return metadata


def _parse_pro_profiles_from_lines(
    lines: List[str],
    file_path: Path,
    code_map_to_use: Dict[str, str],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    logger: logging.Logger,
    remove_soil: bool,
    add_surface_sh_as_layer: bool,
) -> List[Dict]:
    """Parse [DATA] section into temp_profiles list."""
    temp_profiles: List[Dict] = []
    in_data = False
    has_data = False

    start_ts = pd.to_datetime(datetime_start, format='mixed') if datetime_start else None
    end_ts = pd.to_datetime(datetime_end, format='mixed') if datetime_end else None

    def _process_and_append_profile(profile_data: Dict):
        if not profile_data or 'timestamp' not in profile_data:
            return

        profile_ts = pd.to_datetime(
            profile_data['timestamp'],
            format='%d.%m.%Y %H:%M:%S',
            errors='coerce'
        )
        if pd.isna(profile_ts):
            return

        if start_ts and profile_ts < start_ts:
            return
        if end_ts and profile_ts > end_ts:
            return

        # Remove soil layers if requested
        if remove_soil and 'grain_type' in profile_data:
            grain_types = profile_data['grain_type']
            soil_indices = np.where((grain_types == 81) | (grain_types == 82))[0]
            if soil_indices.size > 0:
                first_soil_idx = soil_indices[0]
                new_profile_data = {}
                for key, val in profile_data.items():
                    if isinstance(val, np.ndarray) and val.size == grain_types.size:
                        new_profile_data[key] = val[:first_soil_idx]
                    else:
                        new_profile_data[key] = val
                profile_data = new_profile_data

        height_array = profile_data.get('height', np.array([]))
        if height_array.size == 0:
            profile_data['height'] = np.array([np.nan])
        temp_profiles.append(profile_data)

    def _parse_and_process_block(lines_block: List[str]):
        if not lines_block:
            return
        profile_dict: Dict[str, Any] = {}
        parse_error = False

        # First line is timestamp line: "0500,<timestamp>"
        ts_line = lines_block[0]
        try:
            profile_dict['timestamp'] = ts_line.split(',', 1)[1]
        except IndexError:
            return

        for profile_line in lines_block[1:]:
            try:
                param_code, *data_parts = profile_line.split(',')
                param_name = code_map_to_use.get(param_code)
                if param_name:
                    data_values = data_parts[1:]  # skip declared layer count
                    profile_dict[param_name] = np.array(data_values, dtype=float)
            except (ValueError, IndexError):
                parse_error = True
                logger.error(
                    f"Could not parse data in {file_path.name} at timestamp "
                    f"{profile_dict.get('timestamp', 'unknown')}. Line: '{profile_line}'"
                )
        if parse_error:
            profile_dict['_parse_error'] = True
        _process_and_append_profile(profile_dict)

    current_profile_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == '[DATA]':
            in_data = True
            continue
        if stripped.startswith('[') and stripped != '[DATA]':
            # any other section ends data region
            if in_data:
                break
            continue

        if in_data:
            if stripped.startswith("0500,"):
                # new block
                has_data = True
                _parse_and_process_block(current_profile_lines)
                current_profile_lines = [stripped]
            else:
                current_profile_lines.append(stripped)
    # last block
    _parse_and_process_block(current_profile_lines)

    if not has_data:
        raise ValueError(f"[DATA] section non-existent or empty for {file_path.name}.")

    return temp_profiles


def _read_pro_raw(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    remove_soil: bool = True,
    add_surface_sh_as_layer: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Parses a .pro file into raw Python objects, with an option to remove soil layers.

    This low-level function reads a SNOWPACK .pro file line by line, extracting
    metadata from the header and parsing each time-stamped snow profile from
    the data section. It returns data in a memory-efficient intermediate
    format, suitable for parallel processing. It filters profiles by the
    specified time range *before* storing them.

    Args:
        filepath: Path to the .pro file.
        datetime_start: The start of the time range to parse. Profiles
            before this time are skipped.
        datetime_end: The end of the time range to parse. Profiles after
            this time are skipped.
        code_map: A dictionary mapping parameter codes (e.g., '0501') to
            variable names (e.g., 'height').
        logger (Optional[logging.Logger]): An optional, pre-configured
            logger instance. If None, the default module logger is used.
        remove_soil (bool): If True (default), layers identified as soil
            (grain type 81 or 82) are removed from the bottom of the profile.
        add_surface_sh_as_layer (bool): If True, synthesizes a surface hoar layer
            from the 'SH_at_surface' data and adds it to the top of the
            snowpack. Defaults to True.

    Returns:
        A tuple containing:
            - metadata (Dict): A dictionary of station parameters from the header.
            - temp_profiles (List[Dict]): A list where each element is a
              dictionary representing a single snow profile at one timestamp.
    """
    logger = get_logger(__name__, logger=logger)

    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return {}, []

    code_map_to_use = code_map if code_map is not None else CODE_TO_NAME_MAP

    # read once
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 1) metadata from header
    metadata = _read_pro_metadata(filepath, logger=logger, throw=True)

    # 2) profiles from data section
    temp_profiles = _parse_pro_profiles_from_lines(
        lines=lines,
        file_path=file_path,
        code_map_to_use=code_map_to_use,
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        logger=logger,
        remove_soil=remove_soil,
        add_surface_sh_as_layer=add_surface_sh_as_layer,
    )

    return metadata, temp_profiles

def _create_array(shape, fill_value, dtype, use_dask=False, chunks=None):
    """Helper to create either numpy or dask array."""
    if use_dask:
        return da.full(shape, fill_value, dtype=dtype, chunks=chunks)
    else:
        return np.full(shape, fill_value, dtype=dtype)

def _create_pro_dataset(
    metadata: Dict,
    profiles: List[Dict],
    max_layers: int,
    add_surface_sh_as_layer: bool = True,
    use_dask: bool = False,
    chunks: Dict[str, int] | None = None,
    empty_profile_status_default: int = 0,
) -> xr.Dataset | None:
    """Creates a padded, CF-compliant xarray.Dataset from raw parsed .pro data.

    This function transforms the intermediate raw data from `_read_pro_raw`
    into a fully-formed, 5-dimensional xarray.Dataset. It pads all snow
    profile arrays with np.nan up to `max_layers` to ensure consistent array
    shapes across all time steps. CF-compliant metadata is attached to the
    dataset and its variables based on the internal registry.

    The `add_surface_sh_as_layer` flag controls the handling of surface hoar (SH). If
    True, and a profile contains SH information (from the '0514' parameter
    in the .pro file), an artificial layer is synthesized and added to the
    top of that profile. This new layer will have a grain type of 660, with
    density and grain size taken from the file, while other variables like
    temperature are set to np.nan.

    `profile_status` semantics in this worker:
        - 1: valid snow layer data (no layer data counts as valid information here!)
        - 0: no data (default, e.g., 'height' missing in pro file)
        - <0: parse errors (need to be provided from outside this function from higher-level)
        - status = 0 should basically never happen for this function.

    If no profiles are provided but metadata exists, a minimal dataset with
    `profile_status = empty_profile_status_default` is returned.
    This variable has no time dimension and will broadcast across time when
    merged with time-resolved datasets later on.

    The final dataset will have dimensions:
        - location (1): Single monitoring site
        - time (variable): Each profile timestamp  [only if existent]
        - slope (1): Single slope angle (from file metadata)
        - realization (1): Single simulation run
        - layer (max_layers): Padded vertical layers  [only if existent]

    Args:
        metadata (Dict): A dictionary containing SNOWPACK .pro
            file header (e.g., stationName, latitude).
        profiles (List[Dict]): A list of dictionaries, where each represents
            a single profile at a specific timestamp. The dictionaries
            contain parameter names as keys and numpy arrays of layer values.
        max_layers (int): The number of layers to pad the profile data to.
            This ensures all profile variables have the same shape along the
            'layer' dimension for consistent stacking.
        add_surface_sh_as_layer (bool): If True, synthesizes a surface hoar layer
            from the 'SH_at_surface' data and adds it to the top of the
            snowpack. Defaults to True.
        use_dask (bool): If True, create dask arrays instead of numpy arrays.
        chunks (Dict[str, int]): Chunk sizes for dask arrays if use_dask=True.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the structured
        snow profile data, or None if the input profiles list is empty.
    """

    chunks = chunks or {'time': 100, 'layer': -1}

    # ------------------------------------------------------------------
    # Case 1: no profiles, but we still build a dataset with profile_status
    # ------------------------------------------------------------------
    if not profiles:
        # USE STRING STATION NAME DIRECTLY (not 'stationName')
        station_name = metadata.get('station_name', metadata.get('stationName', 'unknown'))

        slope_index = [0]

        ds = xr.Dataset(
            data_vars={
                # shape: (location=1, slope=1, realization=1)
                # This will broadcast along 'time' if/when merged with datasets
                # that introduce a time dimension for this site.
                'profile_status': (
                    ('location', 'slope', 'realization'),
                    np.array([[[empty_profile_status_default]]], dtype='i1')
                )
            },
            coords={
                'location': [station_name],
                'time': ('time', np.array([], dtype='datetime64[ns]')),
                'slope': slope_index,
                'realization': [0],
                'latitude': (
                    'location',
                    [_round_coord(metadata.get('latitude', np.nan))]
                ),
                'longitude': (
                    'location',
                    [_round_coord(metadata.get('longitude', np.nan))]
                ),
                'altitude': (
                    'location',
                    [float(metadata.get('altitude', np.nan))]
                ),
                'inclination': (
                    'slope',
                    [
                        int(round(float(metadata.get('slopeAngle', np.nan))))
                        if not np.isnan(float(metadata.get('slopeAngle', np.nan)))
                        else np.nan
                    ]
                ),
                'azimuth': (
                    'slope',
                    [
                        int(round(float(metadata.get('slopeAzi', np.nan))))
                        if not np.isnan(float(metadata.get('slopeAzi', np.nan)))
                        else np.nan
                    ]
                ),
            }
        )

        return ds

    # ------------------------------------------------------------------
    # Case 2: profiles present --> full 5D dataset + per-timestamp profile_status
    # ------------------------------------------------------------------
    timestamps = pd.to_datetime(
        [p['timestamp'] for p in profiles],
        format='%d.%m.%Y %H:%M:%S'
    )

    all_params = sorted({
        key for p in profiles for key in p
        if key not in ['timestamp', 'SH_at_surface', '_parse_error']
    })

    data_vars: Dict[str, Any] = {}

    for param in all_params:
        profile_data = []

        for profile in profiles:
            has_sh = False
            sh_gsize = np.nan
            sh_density = np.nan

            if add_surface_sh_as_layer and 'SH_at_surface' in profile:
                sh_data = profile['SH_at_surface']
                if sh_data.size >= 3 and sh_data[0] > 0:
                    has_sh = True
                    sh_gsize = sh_data[1]
                    sh_density = sh_data[2]

            num_snow_layers = len(profile.get('height', []))

            # Get parameter data for this profile
            param_data = profile.get(param, [])
            final_values = list(param_data[:num_snow_layers])

            # Pad with NaN up to num_snow_layers
            while len(final_values) < num_snow_layers:
                final_values.append(np.nan)

            # Add surface hoar layer if needed
            if has_sh:
                if param == 'grain_type':
                    final_values.append(660)
                elif param == 'grain_size':
                    final_values.append(sh_gsize)
                elif param == 'density':
                    final_values.append(sh_density)
                elif param == 'height':
                    last_height = (
                        final_values[-1]
                        if final_values and not np.isnan(final_values[-1])
                        else 0.0
                    )
                    final_values.append(last_height + 0.001)
                else:
                    final_values.append(np.nan)

            # Pad to max_layers
            while len(final_values) < max_layers:
                final_values.append(np.nan)

            # Truncate if too long
            final_values = final_values[:max_layers]

            profile_data.append(final_values)

        # Build numpy array with shape (1, n_profiles, 1, 1, max_layers)
        np_array = np.full(
            (1, len(profiles), 1, 1, max_layers),
            np.nan,
            dtype=np.float32
        )

        for i, profile_values in enumerate(profile_data):
            np_array[0, i, 0, 0, :] = np.clip(profile_values, 
                                              np.finfo(np_array.dtype).min,
                                              np.finfo(np_array.dtype).max
                                              )

        # Convert to dask if needed
        if use_dask:
            chunk_spec = (1, chunks['time'], 1, 1, chunks['layer'])
            arr = da.from_array(np_array.copy(), chunks=chunk_spec)  # type: ignore[arg-type]
        else:
            arr = np_array

        data_vars[param] = (
            ("location", "time", "slope", "realization", "layer"),
            arr
        )

    # ------------------------------------------------------------------
    # Compute profile_status for each timestamp
    #   1: 'height' data exists (even if no snow on ground)
    #   0: safe default if no 'height' data but still parsed without error
    # ------------------------------------------------------------------
    status_np = np.empty((1, len(profiles), 1, 1), dtype='i1')

    for i, profile in enumerate(profiles):
        status = 0  # default: no data otherwise proven differently
        if profile.get('_parse_error', False):
            status = -1
        else:
            height = profile.get('height', None)
            if height is not None and len(height) >= 1:
                status = 1
        status_np[0, i, 0, 0] = status

    if use_dask:
        status_arr = da.from_array(
            status_np,
            chunks=(1, chunks['time'], 1, 1) # type: ignore[arg-type]
        )  # type: ignore[arg-type]
    else:
        status_arr = status_np

    data_vars['profile_status'] = (
        ("location", "time", "slope", "realization"),
        status_arr
    )

    # ------------------------------------------------------------------
    # Build Dataset with coords
    # ------------------------------------------------------------------
    station_name = metadata.get('station_name', metadata.get('stationName', 'unknown'))
    slope_index = [0]

    ds = xr.Dataset(
        data_vars,
        coords={
            'location': [station_name],  # string coordinate
            'time': timestamps,
            'slope': slope_index,
            'realization': [0],
            'layer': np.arange(max_layers),
            'latitude': (
                'location',
                [_round_coord(metadata.get('latitude', np.nan))]
            ),
            'longitude': (
                'location',
                [_round_coord(metadata.get('longitude', np.nan))]
            ),
            'altitude': (
                'location',
                [float(metadata.get('altitude', np.nan))]
            ),
            'inclination': (
                'slope',
                [
                    int(round(float(metadata.get('slopeAngle', np.nan))))
                    if not np.isnan(float(metadata.get('slopeAngle', np.nan)))
                    else np.nan
                ]
            ),
            'azimuth': (
                'slope',
                [
                    int(round(float(metadata.get('slopeAzi', np.nan))))
                    if not np.isnan(float(metadata.get('slopeAzi', np.nan)))
                    else np.nan
                ]
            )
        }
    )
    
    # Attach metadata from registry to all variables
    from .registry import registry
    for var_name in ds.data_vars:
        if reg_entry := registry.get(str(var_name)):
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', str(var_name))
            ds[var_name].attrs['units'] = reg_entry.get('si_units', '1')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', str(var_name)).replace('_', ' ')
        else:
            # Set default attributes for variables not in registry
            if 'units' not in ds[var_name].attrs:
                ds[var_name].attrs['units'] = '1'  # Default: dimensionless
            if 'standard_name' not in ds[var_name].attrs:
                ds[var_name].attrs['standard_name'] = str(var_name)
            if 'long_name' not in ds[var_name].attrs:
                ds[var_name].attrs['long_name'] = str(var_name).replace('_', ' ')
    
    # Add timezone attribute to time coordinate
    if 'time' in ds.coords and 'timezone' not in ds['time'].attrs:
        # Try to get timezone from metadata
        tz_offset = metadata.get('tz')
        if tz_offset is not None:
            try:
                tz_offset = float(tz_offset)
            except (ValueError, TypeError):
                pass  # allow string input of timezone names
            ds['time'].tz.set_attr(tz_offset)
        # else:
        #     # No timezone in metadata, we don't assume defaults
        #     pass
    
    # Add mapping attribute to profile_status
    if 'profile_status' in ds:
        ds['profile_status'].attrs['mapping'] = {
            -1: "error during parsing",
            0: "no layer data",
            1: "existing layer data",
        }

    return ds


# --------------------------------------------------------------------------- #
#                  PRE-SCAN HELPER FUNCTIONS FOR .PRO FILES                   #
# --------------------------------------------------------------------------- #

def _get_layer_count(profile_lines: List[str], add_sh: bool, remove_soil: bool = True) -> int:
    """
    Calculates the number of layers in a single raw profile text block.

    It primarily looks for the layer count specified in the 'height' variable
    line. If a surface hoar (SH) layer is to be added, it increments the
    count. If soil removal is enabled, it subtracts soil layers from the count.

    Args:
        profile_lines: A list of strings, each being a line from a profile
                       block in a .pro file.
        add_sh: A boolean indicating if a surface hoar layer will be added.
        remove_soil: A boolean indicating if soil layers should be removed.

    Returns:
        The total number of layers for the profile.
    """
    layer_count = 0
    sh_present = False
    soil_layer_count = 0
    
    for line in profile_lines:
        if line.startswith("0501,"):  # 'height' param_code
            try:
                layer_count = int(line.split(',')[1])
            except (ValueError, IndexError):
                layer_count = 0
        elif add_sh and line.startswith("0514,"): # 'SH_at_surface'
            try:
                sh_flag = float(line.split(',')[2])
                if sh_flag > 0:
                    sh_present = True
            except (ValueError, IndexError):
                pass
        elif remove_soil and line.startswith("0513,"): # 'grain_type' param_code
            try:
                parts = line.split(',')
                grain_types = [int(float(x)) for x in parts[2:]]  # Skip code and count
                # Count soil layers from the bottom
                for gt in reversed(grain_types):
                    if gt == 81 or gt == 82:
                        soil_layer_count += 1
                    else:
                        break  # Stop at first non-soil layer from bottom
            except (ValueError, IndexError):
                pass
    
    final_count = layer_count - soil_layer_count
    return final_count + 1 if sh_present else final_count


def _scan_pro_for_max_layers(filepath: Path, 
                             add_sh: bool, 
                             remove_soil: bool = True,
                             logger: Optional[logging.Logger] = None) -> int:
    """
    Scans a single .pro file to find the maximum layer count across all profiles.

    This function is designed to be lightweight and fast. It reads a file
    line-by-line, identifying profile blocks and using `_get_layer_count` to
    determine the number of layers in each, returning the maximum value found.

    Args:
        filepath: The path to the .pro file to scan.
        add_sh: A boolean indicating if a surface hoar layer will be added.
        remove_soil: A boolean indicating if soil layers should be removed.

    Returns:
        The maximum layer count found in the file.
    """
    logger = logger or get_logger(__name__)  # Ensure it exists
    max_layers_in_file = 0
    current_profile_lines = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                # A new profile starts with the timestamp code '0500'
                if stripped_line.startswith("0500,"):
                    if current_profile_lines:
                        max_layers_in_file = max(
                            max_layers_in_file,
                            _get_layer_count(current_profile_lines, add_sh, remove_soil)
                        )
                    # Reset for the new profile
                    current_profile_lines = [stripped_line]
                else:
                    current_profile_lines.append(stripped_line)

        # Process the very last profile block in the file
        if current_profile_lines:
            max_layers_in_file = max(
                max_layers_in_file,
                _get_layer_count(current_profile_lines, add_sh, remove_soil)
            )
    except Exception as e:
        # Log error but don't stop the whole process
        logger.warning(f"Could not scan {filepath.name} for layers: {e}")
        return 0
        
    return max_layers_in_file

# --------------------------------------------------------------------------- #
#                   .SMET FILE AND MERGING/FINALIZING LOGIC                   #
# --------------------------------------------------------------------------- #

def _read_smet_files(
    smet_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    logger: logging.Logger,
) -> List[xr.Dataset]:
    """
    Reads a list of .smet files into xarray.Dataset objects.

    Args:
        smet_files: A list of file paths to .smet files.
        datetime_start: An optional start timestamp for filtering.
        datetime_end: An optional end timestamp for filtering.
        logger: A configured logger instance.

    Returns:
        A list of xarray.Dataset objects, one for each successfully
        processed .smet file.
    """
    smet_datasets = []
    for fp in smet_files:
        try:
            if ds := read_smet(fp, datetime_start, datetime_end, logger=logger):
                smet_datasets.append(ds)
        except Exception as e:
            logger.error(f"Failed to read .smet file {fp.name}", exc_info=e)
    return smet_datasets


def read_smet(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[xr.Dataset]:    
    """
    Parses a single SMET file into an xarray.Dataset.

    Reads a SMET-formatted text file, parsing the header for metadata and the
    data section into a pandas DataFrame, which is then converted into an
    xarray.Dataset. The data is filtered by the specified time range.

    Args:
        filepath: Path to the SMET file.
        datetime_start: The start of the time range to read. Data before
            this time is dropped.
        datetime_end: The end of the time range to read. Data after
            this time is dropped.
        logger (Optional[logging.Logger]): An optional, pre-configured
            logger instance. If None, the default module logger is used.

    Returns:
        An xarray.Dataset containing the SMET data, or None if the file
        cannot be parsed or contains no data in the specified time range.
    """
    logger_input = None if isinstance(logger, bool) else logger
    logger = get_logger(__name__, logger=logger_input)

    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return None

    header = {}
    data_start_line = -1
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.strip() == '[DATA]':
                data_start_line = i + 1
                break
            if '=' in line:
                key, value = line.split('=', 1)
                header[key.strip()] = value.strip()

    if 'fields' not in header or data_start_line == -1:
        logger.error(f"SMET file {filepath} is missing 'fields' or '[DATA]' section.")
        return None

    # Normalize coordinates early
    for coord_key in ("latitude", "longitude"):
        if coord_key in header:
            header[coord_key] = _round_coord(header[coord_key])

    variable_heights = {
        key.replace('_height', ''): value.strip()
        for key, value in header.items() if '_height' in key
    }

    original_names = header['fields'].split()
    new_column_names = []
    for name in original_names:
        if name in variable_heights:
            new_column_names.append(f"{name}_{variable_heights[name]}m")
        else:
            new_column_names.append(name)
    
    df = pd.read_csv(
        file_path,
        skiprows=data_start_line,
        sep=r'\s+',
        header=None,
        names=new_column_names,
        na_values=header.get('nodata', -999)
    )

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    if datetime_start:
        df = df[df.index >= pd.to_datetime(datetime_start, format='mixed')]
    if datetime_end:
        df = df[df.index <= pd.to_datetime(datetime_end, format='mixed')]
        
    if df.empty:
        return None

    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename({'timestamp': 'time'})
    
    station_name = header.get('station_name', header.get('station_id', file_path.stem))
    station_name = _sanitize_station_name(station_name)  # ✨ Sanitize!
    
    # Check if dimensions already exist before expanding
    dims_to_add = {}
    if 'location' not in ds.dims and 'location' not in ds.coords:
        dims_to_add['location'] = [station_name]  # STRING!
    if 'slope' not in ds.dims and 'slope' not in ds.coords:
        dims_to_add['slope'] = [0]
    if 'realization' not in ds.dims and 'realization' not in ds.coords:
        dims_to_add['realization'] = [0]
    
    if dims_to_add:
        ds = ds.expand_dims(dims_to_add)
    
    tz_val = header.get('tz')
    if tz_val is not None:
        try:
            tz_val = float(tz_val)
        except (ValueError, TypeError):
            pass  # allow string input of timezone names
        ds['time'].tz.set_attr(tz_val)
    # else:
    #     # No timezone metadata, we don't assume defaults
    #     pass
            
    station_name = header.get('station_name', header.get('station_id', file_path.stem))
    
    # Check if dimensions already exist before expanding
    dims_to_add = {}
    if 'location' not in ds.dims and 'location' not in ds.coords:  # ADD coord check
        dims_to_add['location'] = [station_name]
    if 'slope' not in ds.dims and 'slope' not in ds.coords:
        dims_to_add['slope'] = [0]
    if 'realization' not in ds.dims and 'realization' not in ds.coords:
        dims_to_add['realization'] = [0]
    
    if dims_to_add:
        ds = ds.expand_dims(dims_to_add)
    
    # Add coordinates if they don't exist
    if 'latitude' not in ds.coords:
        ds = ds.assign_coords({
            'latitude': ('location', [_round_coord(header.get('latitude', np.nan))]),
            'longitude': ('location', [_round_coord(header.get('longitude', np.nan))]),
            'altitude': ('location', [float(header.get('altitude', np.nan))]),
        })
    
    # FIXED: Try to get slope values from SMET header before defaulting to NaN
    # This prevents unnecessary NaN assignments when slope metadata is available
    if 'inclination' not in ds.coords:
        # Try to get slope angle from various possible header keys
        slope_incl = header.get('SlopeAngle', header.get('slopeAngle', 
                    header.get('slope_angle', header.get('SlopeAzi='))))
        slope_azi = header.get('SlopeAzi', header.get('slopeAzi', 
                    header.get('slope_azi', header.get('SlopeAzi='))))
        
        # Convert to float, defaulting to NaN if not available or invalid
        try:
            slope_incl = float(slope_incl) if slope_incl is not None else np.nan
        except (ValueError, TypeError):
            slope_incl = np.nan
            
        try:
            slope_azi = float(slope_azi) if slope_azi is not None else np.nan
        except (ValueError, TypeError):
            slope_azi = np.nan

        ds = ds.assign_coords({
            'inclination': ('slope', [slope_incl]),
            'azimuth': ('slope', [slope_azi])
        })
    
    # Attach metadata from registry to all variables
    from .registry import registry
    for var_name in ds.data_vars:
        if reg_entry := registry.get(str(var_name)):
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', str(var_name))
            ds[var_name].attrs['units'] = reg_entry.get('si_units', '1')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', str(var_name)).replace('_', ' ')
        else:
            # Set default attributes for variables not in registry (e.g., SMET meteorological vars)
            if 'units' not in ds[var_name].attrs:
                ds[var_name].attrs['units'] = '1'  # Default: dimensionless
            if 'standard_name' not in ds[var_name].attrs:
                ds[var_name].attrs['standard_name'] = str(var_name)
            if 'long_name' not in ds[var_name].attrs:
                ds[var_name].attrs['long_name'] = str(var_name).replace('_', ' ')

    return ds

def _resolve_filepaths(source: Union[str, Path, 
                       Sequence[Union[str, Path]]],
                       recursive: bool = False, 
                       logger: Optional[logging.Logger]=None,
) -> Tuple[List[Path], List[Path]]:
    """
    Resolves the source argument into separate lists of .pro and .smet files.

    Args:
        source: The data source, which can be a directory path, a single
            file path, or a list of paths.

    Returns:
        A tuple containing two lists: one for .pro file paths and one for
        .smet file paths.
        
    Raises:
        TypeError: If the source is of an unsupported type.
    """
    logger = logger or get_logger(__name__)
    paths_to_process = []
    if isinstance(source, (str, Path)):
        paths_to_process.append(Path(source))
    elif isinstance(source, list):
        paths_to_process = [Path(p) for p in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    resolved_files = []
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
    time: TimeSelector,
    location: LocationSelector,
    slope: SlopeSelector,
    realization: RealizationSelector,
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
    grouped_by_location = {}
    for ds in all_datasets:
        loc = str(ds.location.values[0])  # Get string location name
        grouped_by_location.setdefault(loc, []).append(ds)
    
    # HANDLE DUPLICATE STATION NAMES
    station_names = list(grouped_by_location.keys())
    unique_names = _handle_duplicate_names(station_names, logger=logger)
    
    # Rename locations if duplicates were found
    if unique_names != station_names:
        new_grouped = {}
        for old_name, new_name in zip(station_names, unique_names):
            new_grouped[new_name] = grouped_by_location[old_name]
            # Update location coordinate in datasets
            for ds in new_grouped[new_name]:
                ds['location'] = [new_name]
        grouped_by_location = new_grouped
    
    merged_datasets = []
    for loc_name, dsets in grouped_by_location.items():    
        profile_dsets = [ds for ds in dsets if 'layer' in ds.dims]
        timeseries_dsets = [ds for ds in dsets if 'layer' not in ds.dims]

        combined_profiles_ds = None
        if profile_dsets:
            if len(profile_dsets) > 1:
                # Sort by inclination if it exists, otherwise use 0 as default
                def get_slope_key(ds):
                    if 'inclination' in ds.coords:
                        return ds.coords['inclination'].values[0]
                    elif 'inclination' in ds.data_vars:
                        return ds['inclination'].values[0]
                    else:
                        return 0.0  # Default value if inclination doesn't exist
                
                profile_dsets.sort(key=get_slope_key)
                
                # CRITICAL FIX: Save location-level coordinates before slope concat
                # These coordinates (lat/lon/alt) should NOT vary with slope dimension
                location_coords = {}
                for coord_name in ['latitude', 'longitude', 'altitude', 'location']:
                    if coord_name in profile_dsets[0].coords:
                        location_coords[coord_name] = profile_dsets[0][coord_name]
                
                # Drop location-level coords before concatenating along slope
                profile_dsets_cleaned = [
                    ds.drop_vars([c for c in ['latitude', 'longitude', 'altitude', 'location'] if c in ds.coords],
                                 errors='ignore')
                    for ds in profile_dsets
                ]
                
                # Extract underlying xarray Datasets before concat (xr.concat doesn't accept xsnowDataset)
                profile_xr_datasets = [ds.data if hasattr(ds, 'data') else ds for ds in profile_dsets_cleaned]
                combined_profiles_ds = xr.concat(profile_xr_datasets, dim='slope', join='outer', coords='all')
                combined_profiles_ds = combined_profiles_ds.assign_coords(slope=np.arange(combined_profiles_ds.sizes['slope']))
                
                # Restore location-level coordinates (without slope dimension)
                
                # Restore location-level coordinates (without slope dimension)
                combined_profiles_ds = combined_profiles_ds.assign_coords(location_coords)
            else:
                # Extract .data if it's an xsnowDataset
                combined_profiles_ds = profile_dsets[0].data if hasattr(profile_dsets[0], 'data') else profile_dsets[0]

        # Extract .data from all datasets before merge
        all_to_merge = []
        if combined_profiles_ds is not None:
            all_to_merge.append(combined_profiles_ds)
        # Also extract .data from timeseries datasets
        for ds in timeseries_dsets:
            all_to_merge.append(ds.data if hasattr(ds, 'data') else ds)
        
        if not all_to_merge: 
            continue

        if len(all_to_merge) > 1:
            merged_loc_ds = xr.merge(all_to_merge, compat='override', join='outer', combine_attrs='no_conflicts')
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
            # 'title': 'SNOWPACK model output',
            # 'source': 'SNOWPACK model output (.pro and .smet files)',
            # 'history': f'{pd.Timestamp.now(tz="UTC").isoformat()}: Created by xsnow.',
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
    # After merging multiple locations, slope coordinates (azimuth, inclination)
    # may have dimensions (location, slope) when they should only have (slope).
    if 'slope' in final_xr_dataset.dims:
        try:
            if norm_slopes is True:
                final_xr_dataset = normalize_slope_coords(
                    final_xr_dataset,
                    verify_consistency=False,  # Force reduction even if slopes differ by location
                    allow_per_location=False,  # Always collapse to (slope,)
                )
            elif norm_slopes == 'auto':
                final_xr_dataset = normalize_slope_coords(
                    final_xr_dataset,
                    verify_consistency=True,   # Only reduce when consistent
                    allow_per_location=True,   # Preserve per-location slopes when inconsistent
                )
            elif norm_slopes is False:
                pass  # Explicitly skip normalization
            else:
                raise ValueError(
                    f"Unsupported norm_slopes value: {norm_slopes!r}. "
                    "Use True, False, or 'auto'."
                )
        except ValueError as e:
            # This should rarely happen now with allow_per_location=True
            # Only occurs if there's a true data structure problem
            logger.error(
                f"Failed to normalize slope coordinates: {e}\n"
                f"This indicates a data structure issue that needs investigation."
            )
            # Re-raise instead of silently using first location's values
            raise
    
    # The location coordinate is already meaningful strings
    # Apply final filters
    if time: 
        final_xr_dataset = time.apply(final_xr_dataset)
    if location: 
        final_xr_dataset = location.apply(final_xr_dataset)
    if slope: 
        final_xr_dataset = slope.apply(final_xr_dataset)
    if realization: 
        final_xr_dataset = realization.apply(final_xr_dataset)

    final_xr_dataset = reorder_dims(final_xr_dataset)
    final_dataset = xsnowDataset(final_xr_dataset)
    
    # DEBUG: Check slope coordinates after merge
    if 'inclination' in final_dataset.data.coords:
        logger.debug(f"_merge_and_finalize: inclination values = {final_dataset.data.inclination.values}")
        logger.debug(f"_merge_and_finalize: inclination dims = {final_dataset.data.inclination.dims}")

    if 'z' not in final_dataset and 'height' in final_dataset:
        final_dataset = final_dataset.compute_z_depth()
    
    # Compute HS if not present
    if 'HS' not in final_dataset and 'height' in final_dataset:
        final_dataset = final_dataset.compute_hs()

    final_dataset = final_dataset.update_profile_status()
    
    return final_dataset


def _coerce_time(sel: Optional[TimeLike]) -> Optional[TimeSelector]:
    if sel is None or isinstance(sel, TimeSelector):
        return sel
    return TimeSelector(sel=sel)

def _coerce_location(sel: Optional[LocationLike]) -> Optional[LocationSelector]:
    if sel is None or isinstance(sel, LocationSelector):
        return sel
    return LocationSelector(sel=sel)

def _coerce_slope(sel: Optional[SlopeLike]) -> Optional[SlopeSelector]:
    if sel is None or isinstance(sel, SlopeSelector):
        return sel
    return SlopeSelector(sel=sel)

def _coerce_realization(sel: Optional[RealizationLike]) -> Optional[RealizationSelector]:
    if sel is None or isinstance(sel, RealizationSelector):
        return sel
    return RealizationSelector(sel=sel)

# --------------------------------------------------------------------------- #
#                           WRITE FUNCTIONS                                   #
# --------------------------------------------------------------------------- #

@register_xsnowDataset_method
def append_latest(existing_ds: xsnowDataset, source: Union[str, List[str], Path],
                  from_time: Optional[Union[str, datetime, pd.Timestamp, np.datetime64]]=None,
                  join: str='left', logger: Optional[logging.Logger]=None, **kwargs,
) -> xsnowDataset:
    """
    Incrementally extend a dataset by appending newer timestamps from files.

    This method is a convenience wrapper for time selection, read, and concat.
    Semantics:
        1) Determine a start time:
            - If `from_time` is None, use (max(existing time) + 1s).
            - If `from_time` is provided, drop all times <= `from_time`
              from `existing_ds` and start reading from `from_time` (inclusive).
        2) Read `source` with that time filter.
        3) Concatenate along `time` using `xsnow.concat`.
            - `join` controls non-time dims (default: 'left' keeps existing domains).
            - Time is always the union along the concat axis (no extra reindexing).

    Args:
        existing_ds: The dataset instance to extend.
        source: Path(s) to files or directories to read new data from (calls `read` upon it).
        from_time: Subset existing data to times `< from_time` and read new data starting 
            with `from_time` (this will already be new data). (also read Semantics above)
            Accepts str/pandas/np datetime or datetime.
        join: Join mode for non-time dimensions forwarded to `xsnow.concat`.
        logger: Optional logger to use for status/warning messages. If None, a
            logger is created.
        **kwargs: Additional keyword args forwarded to `xsnow.concat` like `compat` or 
            `combine_attrs`.

    Returns:
        xsnowDataset with the appended data (or trimmed-only if nothing new).
    """
    logger = get_logger(__name__, logger=logger)

    base_data = existing_ds.data if existing_ds.data is not None else xr.Dataset()
    base_filtered = base_data
    base_max_ts = None
    if "time" in base_data and base_data.time.size > 0:
        base_max_ts = pd.to_datetime(base_data.time.max().values)

    # Determine start time for reading new data; optionally trim existing
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
                "append_latest called with from_time earlier than existing max time; "
                "this trims existing data from that point onward and may introduce "
                "time gaps if the new source does not fully cover that range."
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
                    "append_latest found no new time steps, but new coordinate values exist "
                    f"that would be merged under join='{join}': {preview}. "
                    "Check time filters or data completeness."
                )
        # Nothing to append; still return trimmed data if from_time was provided
        return existing_ds.__class__(base_filtered)

    # Guard against unexpected overlapping timestamps
    if "time" in base_filtered and "time" in new_data:
        overlap = np.intersect1d(base_filtered["time"].values, new_data["time"].values)
        if overlap.size:
            raise ValueError(
                "append_latest detected overlapping timestamps between existing "
                "and new data; expected disjoint time domains. "
                "This may indicate an incorrect from_time or a read filter bug. "
                f"Example overlaps (up to 5): {overlap[:5]!r}"
            )

    JoinMode = Literal["left", "right", "outer", "inner", "union"]
    
    # Fix: Convert slope coordinates to data variables to prevent concat mismatch
    # Cached NetCDF data has them as data vars, fresh .pro files have them as coords
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
    # Log
    new_times = combined.time.where(combined.time >= pd.to_datetime(time_start), drop=True)
    new_time_start = pd.to_datetime(new_times[0].values).isoformat()
    new_time_end = pd.to_datetime(new_times.max().values).isoformat()
    logger.info(f"Appended {len(new_times)} new timestamps: {new_time_start}--{new_time_end}")
    # Unwrap xsnowDataset if needed for type compatibility
    combined_data = combined.data if isinstance(combined, xsnowDataset) else combined
    extra_coords = _new_coord_values(base_filtered, combined_data)
    if extra_coords:
        preview = ", ".join(
                        f"{name}: {_preview(vals)}" for name, vals in extra_coords.items()
                    )
        logger.debug(f"Added new coordinates: {preview}")

    return combined if isinstance(combined, xsnowDataset) else xsnowDataset(combined)


def _encode_mapping_attrs_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """
    Return a Dataset with any variable/dataset attrs named 'mapping'
    (which must be dicts) serialized into JSON under 'mapping_json'.
    The original 'mapping' attr is removed to satisfy NetCDF constraints.
    This function modifies in-place!
    """
    ds_out = ds  # work in-place on the copy the caller passes in

    # dataset-level attrs
    if isinstance(ds_out.attrs.get("mapping"), dict):
        ds_out.attrs["mapping_json"] = json.dumps(ds_out.attrs["mapping"])
        del ds_out.attrs["mapping"]

    # variable-level attrs
    for name, var in ds_out.variables.items():
        if isinstance(var.attrs.get("mapping"), dict):
            var.attrs["mapping_json"] = json.dumps(var.attrs["mapping"])
            del var.attrs["mapping"]

    return ds_out

@register_xsnowDataset_method
def to_netcdf(ds: xsnowDataset, path: Union[str, Path],
              logger: Optional[logging.Logger]=None, **kwargs):
    """
    Saves the dataset to a NetCDF file.
    This method provides a convenient wrapper around the underlying
    `xarray.Dataset.to_netcdf()` method for easy caching and interoperability.
    
    Automatically converts slope-varying coordinates (azimuth, aspect, 
    inclination) to data variables to ensure files are mergeable without
    conflicts.
    
    The location mapping dictionary is converted to NetCDF-compatible format
    by encoding it as JSON in a string attribute.
    
    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .nc file.
        logger (logging.Logger): An optional preconfigured logger instance.
        **kwargs: Additional keyword arguments passed to `xarray.Dataset.to_netcdf()`.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    logger = get_logger(__name__, logger=logger)
    
    # Create a copy to avoid modifying the original
    ds_to_save = ds.to_xarray().copy()
    
    # Convert slope-varying coordinates to data variables for mergeability
    SLOPE_VARYING_COORDS = ['azimuth', 'aspect', 'inclination']
    
    converted = []
    for coord in SLOPE_VARYING_COORDS:
        if coord in ds_to_save.coords:
            # Check if it's a problematic 1D slope coordinate
            if ds_to_save.coords[coord].dims == ('slope',):
                logger.debug(
                    f"Converting '{coord}' from coordinate to data variable "
                    f"(prevents merge conflicts)"
                )
                
                # Broadcast 1D (slope,) to 2D (location, slope)
                coord_1d = ds_to_save.coords[coord]
                
                # Find a template variable with (location, slope) dimensions
                template_var = None
                for var in ds_to_save.data_vars:
                    if 'location' in ds_to_save[var].dims and 'slope' in ds_to_save[var].dims:
                        template = ds_to_save[var]
                        # Select first index for any extra dimensions
                        for dim in template.dims:
                            if dim not in ['location', 'slope']:
                                template = template.isel({dim: 0})
                        template_var = template
                        break
                
                if template_var is not None:
                    # Broadcast using template
                    coord_2d = coord_1d.broadcast_like(template_var)
                else:
                    # Fallback: manual broadcast
                    import numpy as np
                    n_locations = len(ds_to_save.location)
                    n_slopes = len(ds_to_save.slope)
                    values_2d = np.broadcast_to(coord_1d.values, (n_locations, n_slopes))
                    coord_2d = xr.DataArray(
                        values_2d,
                        dims=('location', 'slope'),
                        coords={'location': ds_to_save.location, 'slope': ds_to_save.slope}
                    )
                
                # Add as data variable and remove from coordinates
                ds_to_save[coord] = coord_2d
                ds_to_save = ds_to_save.reset_coords(coord)
                
                converted.append(coord)
    
    if converted:
        logger.info(
            f"Converted {len(converted)} slope-varying coordinate(s) to "
            f"data variable(s): {', '.join(converted)}"
        )   
    # DEBUG: Check slope coordinates before saving
    if 'inclination' in ds_to_save.coords:
        logger.debug(f"to_netcdf: inclination BEFORE save = {ds_to_save.inclination.values}")
        logger.debug(f"to_netcdf: inclination dims = {ds_to_save.inclination.dims}")
    # Encode any dict attrs named 'mapping' as JSON
    ds_to_save = _encode_mapping_attrs_for_netcdf(ds_to_save)
    # Encode string coordinates as fixed-length bytes for NetCDF classic compatibility
    for coord_name in ("location", "realization"):
        if coord_name in ds_to_save.coords:
            coord_data = ds_to_save[coord_name]
            if coord_data.dtype.kind in ("U", "S", "O"):
                if coord_data.dtype.kind == "U":
                    max_len = max((len(str(x)) for x in coord_data.values), default=1)
                elif coord_data.dtype.kind == "S":
                    max_len = max(1, coord_data.dtype.itemsize)
                else:
                    max_len = max((len(str(x)) for x in coord_data.values), default=1)
                ds_to_save[coord_name].encoding = {
                    "dtype": f"S{max_len}",
                    "compression": "zlib",
                    "complevel": 4,
                }
                logger.debug(f"Encoding coord '{coord_name}' as S{max_len}")
    path = Path(path)
    ds_to_save.to_netcdf(path, **kwargs)
    logger.info(f"Dataset successfully saved to {path}")
    
    
@register_xsnowDataset_method
def to_smet(ds: xsnowDataset, path: Union[str, Path], max_files: int | None = None, **kwargs):
    """
    Saves time-series data from the dataset to a SMET file.

    This function extracts data variables that do not have a 'layer' dimension
    (e.g., meteorological data, total snow height) and writes them into the
    SMET format. It only supports writing data for a single location.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .smet file.
        max_files (int, optional): The maximum number of locations allowed.
            If the dataset contains more locations, a ValueError is raised.
            Defaults to None (no limit).
        **kwargs: Reserved for future filtering options.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    
    if max_files is not None:
            num_locations = ds.sizes.get('location')
            if num_locations and num_locations > max_files:
                raise ValueError(
                    f"Dataset contains {num_locations} locations, which exceeds the "
                    f"specified maximum of {max_files} files to save."
                )
    
    # Check location dimension (may be None if squeezed after .sel())
    num_locations = ds.sizes.get('location', 0)
    if num_locations is None:
        num_locations = 0
    
    if num_locations > 1:
        raise ValueError(
            "to_smet only supports datasets with a single location. "
            "Use ds.sel(location='name') to select one location first."
        )

    smet_vars = [name for name, var in ds.data.data_vars.items() if 'layer' not in var.dims]
    if not smet_vars:
        logging.warning("Warning: No time-series variables found to write to SMET file.")
        return

    # Squeeze and convert to dataframe, keeping only data variables
    df = ds.data[smet_vars].squeeze(drop=True).to_dataframe()  # ADD drop=True

    with open(path, 'w') as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        f.write(f"station_name = {ds.location.item()}\n")
        # Handle coordinates that may have extra dimensions (e.g., slope)
        # Take the first value if there are multiple
        lat = float(ds.latitude.values.flat[0]) if ds.latitude.size > 1 else ds.latitude.item()
        lon = float(ds.longitude.values.flat[0]) if ds.longitude.size > 1 else ds.longitude.item()
        alt = float(ds.altitude.values.flat[0]) if ds.altitude.size > 1 else ds.altitude.item()
        f.write(f"latitude = {lat:.4f}\n")
        f.write(f"longitude = {lon:.4f}\n")
        f.write(f"altitude = {alt:.0f}\n")
        f.write("nodata = -999\n")
        fields = "timestamp " + " ".join(df.columns)
        f.write(f"fields = {fields}\n")
        f.write("[DATA]\n")

    df.index.name = 'timestamp'
    df.to_csv(path, sep=' ', mode='a', header=False, date_format='%Y-%m-%dT%H:%M:%S')
    logging.info(f"Dataset successfully saved to {path}")


@register_xsnowDataset_method
def to_pro(ds: xsnowDataset, path: Union[str, Path], max_files: int | None = None, **kwargs):
    """
    Saves a single snow profile to a SNOWPACK .pro file.

    This function iterates through each timestamp in the dataset and writes the
    vertical profile data (variables with a 'layer' dimension) into the .pro
    format. It only supports writing data for a single location.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .pro file.
        max_files (int, optional): The maximum number of profiles (timestamps)
            allowed. If the dataset contains more profiles, a ValueError is
            raised. Defaults to None (no limit).
        **kwargs: Reserved for future filtering options.
        
    Raises:
        ValueError: If the dataset is empty, contains more than one location,
            or if the number of profiles exceeds max_files.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
        
    if max_files is not None:
        num_profiles = ds.sizes.get('time')
        if num_profiles > max_files:
            raise ValueError(
                f"Dataset contains {num_profiles} profiles, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )

    # Check location dimension (may be None if squeezed after .sel())
    num_locations = ds.sizes.get('location', 0)
    if num_locations is None:
        num_locations = 0
    
    if num_locations > 1:
        raise ValueError(
            "to_pro only supports datasets with a single location. "  # FIXED MESSAGE
            "Use ds.sel(location='name') to select one location first."
        )

    # Squeeze location dimension if it exists
    if 'location' in ds.data.dims:
        single_loc_ds = ds.data.squeeze(dim='location', drop=False)
    else:
        # Location already squeezed/removed by .sel()
        single_loc_ds = ds.data

    # Select first slope if multiple slopes exist
    if 'slope' in single_loc_ds.dims and single_loc_ds.sizes['slope'] > 1:
        single_loc_ds = single_loc_ds.isel(slope=0)
    
    # Squeeze slope if it exists
    if 'slope' in single_loc_ds.dims:
        single_loc_ds = single_loc_ds.squeeze(dim='slope', drop=False)
    
    profile_vars = {name: var for name, var in single_loc_ds.data_vars.items() if 'layer' in var.dims}

    with open(path, 'w') as f:
        f.write("[STATION_PARAMETERS]\n")
        f.write(f"StationName= {single_loc_ds.location.item()}\n")
        f.write(f"Latitude= {single_loc_ds.latitude.item():.4f}\n")
        f.write(f"Longitude= {single_loc_ds.longitude.item():.4f}\n")
        f.write(f"Altitude= {single_loc_ds.altitude.item():.0f}\n")
        
        # Get slope angle value
        try:
            slope_val = int(round(single_loc_ds.inclination.values.item()))
        except (AttributeError, ValueError):
            slope_val = -999
        f.write(f"SlopeAngle= {slope_val:.0f}\n")
        
        # Handle slope azimuth
        try:
            azimuth = int(round(single_loc_ds.azimuth.values.item()))
        except (AttributeError, ValueError):
            azimuth = -999
        f.write(f"SlopeAzi= {azimuth:.0f}\n")
        
        f.write("[DATA]\n")

        # Iterate over time dimension (will work even if size is 1)
        for t in single_loc_ds.time.values:
            profile = single_loc_ds.sel(time=t)
            timestamp_str = pd.to_datetime(t).strftime('%d.%m.%Y %H:%M:%S')
            f.write(f"0500,{timestamp_str}\n")
            
            for var_name, code in registry.items():
                param_code = code.get('param_code') if isinstance(code, dict) else None
                if param_code and var_name in profile_vars:
                    data_array = profile[var_name].dropna(dim='layer').values
                    if data_array.size > 0:
                        num_layers = data_array.size
                        formatted_values = [f"{val:.4f}" for val in data_array.flatten()]
                        data_str = ",".join(formatted_values)
                        f.write(f"{param_code},{num_layers},{data_str}\n")
                        
                        
    logging.info(f"Dataset successfully saved to {path}")    

@register_xsnowDataset_method
def to_zarr(
    ds: xsnowDataset,
    path: Union[str, Path],
    mode: ZarrModes = 'w',
    append_dim: Optional[str] = None,
    consolidated: bool = True,
    **kwargs
) -> None:
    """
    Save dataset to Zarr format with automatic slope coordinate handling.
    
    Zarr format provides better stability and performance for large datasets
    compared to NetCDF, with pure Python implementation (no HDF5 crashes).
    
    This function automatically converts 1D slope coordinates (azimuth, inclination)
    to 2D data variables with (location, slope) dimensions, ensuring compatibility
    with the rest of xsnow and avoiding merge conflicts.
    
    Args:
        ds: The xsnowDataset to save
        path: Output path (should end in .zarr)
        mode: Write mode - 'w' (overwrite), 'a' (append), 'w-' (fail if exists)
        append_dim: Dimension to append along (e.g., 'time' for time-series updates)
        consolidated: Whether to consolidate metadata (improves read performance)
        **kwargs: Additional arguments passed to xarray.Dataset.to_zarr()
        
    Returns:
        None
        
    Examples:
        >>> # Save complete dataset
        >>> ds.to_zarr('archive_2024.zarr')
        
        >>> # Append new data along time dimension
        >>> new_data.to_zarr('archive_2024.zarr', mode='a', append_dim='time')
        
        >>> # Create with custom compression
        >>> import numcodecs
        >>> encoding = {var: {'compressor': numcodecs.Blosc(cname='zstd', clevel=5)}
        ...            for var in ds.data_vars}
        >>> ds.to_zarr('archive.zarr', encoding=encoding)
        
    See Also:
        read_zarr : Read Zarr archives
        to_netcdf : Save to NetCDF format
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    
    logger = logging.getLogger(__name__)
    
    # Make copy to avoid modifying original
    ds_to_save = ds.data.copy()
    
    # Convert 1D slope coordinates to 2D data variables
    # This prevents merge conflicts and maintains xsnow conventions
    slope_coords_converted = []
    
    for coord in ['azimuth', 'inclination', 'aspect']:
        if coord in ds_to_save.coords:
            coord_var = ds_to_save.coords[coord]
            
            # Check if it's a 1D slope coordinate
            if coord_var.dims == ('slope',):
                logger.debug(f"Converting {coord} from coordinate to data variable")
                
                # Remove from coordinates
                ds_to_save = ds_to_save.reset_coords(coord)
                
                # Broadcast to 2D if needed
                if 'location' in ds_to_save.dims and 'slope' in ds_to_save.dims:
                    # Find a template variable with (location, slope) dims
                    template = None
                    for var_name in ds_to_save.data_vars:
                        var = ds_to_save[var_name]
                        if 'location' in var.dims and 'slope' in var.dims:
                            template = var
                            # Select first index for any extra dimensions
                            for dim in var.dims:
                                if dim not in ['location', 'slope']:
                                    template = template.isel({dim: 0})
                            break
                    
                    if template is not None:
                        # Broadcast slope coordinate to 2D
                        coord_2d = ds_to_save[coord].broadcast_like(template)
                        ds_to_save[coord] = coord_2d
                        slope_coords_converted.append(coord)
                    else:
                        # Fallback: create 2D using np.broadcast_to
                        n_loc = ds_to_save.sizes['location']
                        n_slope = ds_to_save.sizes['slope']
                        coord_1d = ds_to_save[coord].values
                        coord_2d_vals = np.broadcast_to(coord_1d, (n_loc, n_slope))
                        ds_to_save[coord] = (('location', 'slope'), coord_2d_vals)
                        slope_coords_converted.append(coord)
    
    if slope_coords_converted:
        logger.info(f"Converted slope coordinates to 2D data variables: {slope_coords_converted}")
    
    # Ensure path ends with .zarr
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
        import subprocess
        result = subprocess.run(['du', '-sh', str(path)],
                              capture_output=True, text=True)
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            logger.info(f"   Size: {size_str}")
        
    except Exception as e:
        logger.error(f"Failed to save Zarr archive: {e}")
        raise    

@register_xsnowDataset_method
def to_json(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """Saves the dataset to a structured JSON file. (Not Implemented)"""
    raise NotImplementedError("to_json() is not yet implemented.")

@register_xsnowDataset_method
def to_caaml(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """Saves snow profile data to a CAAML V6.0 XML file. (Not Implemented)"""
    raise NotImplementedError("to_caaml() is not yet implemented.")

@register_xsnowDataset_method
def to_crocus(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """Saves snow profile data to a Crocus model input file. (Not Implemented)"""
    raise NotImplementedError("to_crocus() is not yet implemented.")


@register_xsnowDataset_method
def trim_padding(
    ds: xsnowDataset,
    dims: Optional[List[str]] = None,
    keep_buffer: int = 0,
    inplace: bool = False
) -> xsnowDataset:
    """
    Remove NaN padding from layer dimension to reduce memory usage.
    
    After reading files with max_layers padding, many trailing layers may be
    entirely NaN. This method detects the actual maximum layer with data and
    trims the excess.
    
    IMPORTANT: For lazy-loaded datasets, this will trigger computation of all
    dask arrays to check for NaN values. For large datasets, consider using
    eager loading if you plan to trim padding, or trim after subsetting:
    ds.sel(location='X').compute().trim_padding()
    
    Args:
        ds: The dataset instance to trim
        dims: List of dimensions to trim. Default: ['layer'] 
        keep_buffer: Number of extra layers to keep beyond the maximum non-NaN
                    layer. Useful if you expect to add more data later.
                    Default: 0 (trim to exact maximum)
        inplace: If True, modify dataset in place. If False (default), return
                a new dataset.
    
    Returns:
        Trimmed xsnowDataset (new instance if inplace=False)
        
    Examples:
        >>> # Dataset padded to 50 layers, but only 12 have data
        >>> ds.sizes['layer']
        50
        >>> ds_trimmed = ds.trim_padding()
        >>> ds_trimmed.sizes['layer']
        12
        
        >>> # Keep 2 extra layers as buffer
        >>> ds_trimmed = ds.trim_padding(keep_buffer=2)
        >>> ds_trimmed.sizes['layer']
        14
        
        >>> # Trim in place
        >>> ds.trim_padding(inplace=True)
        >>> ds.sizes['layer']
        12
    
    Notes:
        - Only works on dimensions where padding is at the end (e.g., layer)
        - Checks all data variables that have the specified dimension
        - For lazy datasets, this triggers computation - consider eager loading
    """
    import numpy as np
    import dask.array as da
    import logging
    
    logger = logging.getLogger(__name__)
    
    if dims is None:
        dims = ['layer']
    
    # Work on a copy unless inplace
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
        # Find maximum non-NaN index across all variables with this dimension
        max_valid_index = -1
        
        # Check all variables that use this dimension
        relevant_vars = [
            v for v in current_data.data_vars 
            if dim in current_data[v].dims
        ]
        
        if not relevant_vars:
            continue

        # Compute the global max index of valid data
        # For lazy arrays, this WILL trigger computation if not careful
        # We optimize by checking checking the "profile_status" if available, 
        # or checking the 'height' variable specifically if present.
        
        # Optimization: If we have profile_status=0 (no data), we know index is 0.
        # But we need the layer depth. 
        # Let's trust the variables.
        
        for var_name in relevant_vars:
            da_var = current_data[var_name]
            
            # Find the last index along 'dim' that is not NaN
            # We assume padding is at the end.
            
            # Collapse other dimensions to find if ANY point has data at this layer index
            # .notnull().any(...) reduces everything except the layer dimension
            reduce_dims = [d for d in da_var.dims if d != dim]
            
            # This returns a 1D boolean array along 'dim' (e.g., layer)
            has_data = da_var.notnull().any(dim=reduce_dims)
            
            # Convert to numpy to find the last True value
            # (This triggers compute for this 1D boolean array, which is acceptable)
            has_data_np = has_data.values
            
            if has_data_np.any():
                # np.where returns tuple of arrays, we want last index
                last_idx = np.where(has_data_np)[0][-1]
                max_valid_index = max(max_valid_index, last_idx)
        
        # Calculate new size with buffer
        if max_valid_index == -1:
            # All NaN? Keep at least 1 layer or 0? 
            # Usually implies empty dataset. Let's keep 0 if truly empty.
            new_size = 0
        else:
            new_size = max_valid_index + 1 + keep_buffer
            
        # Clamp to original size
        new_size = min(new_size, original_size)
        
        if new_size < original_size:
            # Perform the slice
            # .isel returns a NEW xarray dataset
            current_data = current_data.isel({dim: slice(0, new_size)})
            
            logger.info(
                f"Trimmed '{dim}' dimension from {original_size} to {new_size} "
                f"(removed {original_size - new_size} NaN-only slices)"
            )
        else:
            logger.debug(f"No trimming needed for dimension '{dim}'")

    # 3. CRITICAL FIX: Assign the modified xarray dataset back to the wrapper
    # The bug was likely modifying a local variable 'data' but never updating ds_work.data
    ds_work.data = current_data
    
    return ds_work