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
import os
import sys
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, List, Any, Union, Tuple, Sequence, Mapping
import concurrent.futures


import numpy as np
import pandas as pd
import xarray as xr
from dask.base import compute
from dask.delayed import delayed
import dask.array as da

# Import from the package's modules
from .registry import registry
from .core import xsnowDataset
from .selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from .plugin import register_xsnowDataset_method

# Configure a basic logger for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'StationName=': 'stationName'
}

# --- Selector Types ---
IndexLike = Union[int, slice, Sequence[int]]

TimeIndexScalar = Union[str, pd.Timestamp, np.datetime64]
TimeIndexSeq    = Sequence[TimeIndexScalar]
TimeSlice       = slice
TimeLike = Union["TimeSelector", TimeIndexScalar, TimeSlice, TimeIndexSeq]

LocationIndexScalar = str
LocationIndexSeq    = Sequence[LocationIndexScalar]
# - NOTE: bbox args (lat_range/lon_range/crs) cannot be expressed
#   positionally here, so callers that want bbox still need to pass
#   an actual LocationSelector(...)
LocationLike = Union["LocationSelector", LocationIndexScalar, LocationIndexSeq]

SlopeIndexLike = Union[int, slice, Sequence[int], xr.DataArray]  # xr <-> boolean mask
SlopeQueryMapping = Mapping[str, Any]  # e.g., {"inclination": 30, "azimuth": 180}
SlopeLike = Union["SlopeSelector", SlopeIndexLike, SlopeQueryMapping]

RealIndexLike = Union[int, str, slice, Sequence[int], Sequence[str]]
RealizationLike = Union["RealizationSelector", RealIndexLike]

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
            (default), auto-detect based on file count (>10 files uses lazy).
        n_cpus_use: The number of CPU cores to use. If not provided, defaults
            to min(32, total_cores - 1), with a minimum of 1.
        logger: An optional, pre-configured logger instance.
        chunks: Chunk sizes for dask arrays when lazy=True.       
                Dict with keys 'time' and 'layer'.                  
                Default: {'time': 100, 'layer': -1}                
                Example: {'time': 50, 'layer': 100}                
        **parser_kwargs: Additional keyword arguments passed to the underlying
            parser functions, such as `max_layers='auto'`, `remove_soil=True`,
            `add_surface_sh_as_layer=True`.

    Returns:
        A unified `xsnowDataset` object containing the data from all sources,
        or `None` if no valid data files are found.

    Examples:
        >>> # Auto-detect (lazy for >10 files, eager for <=10)
        >>> ds = read("data/")
        
        >>> # Force lazy loading for large datasets
        >>> ds = read("large_archive/", lazy=True, n_cpus_use=8)
        
        >>> # Force eager loading for quick access
        >>> ds = read("small_dataset/", lazy=False, n_cpus_use=4)
    """
# --- Step 1: Setup ---
    if logger is None or logger is True:
        logger = logging.getLogger(__name__)

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
            # Using context managers to ensure file is properly closed
            with xr.open_dataset(source) as ds:
                ds = ds.load()

            # Decode bytes to strings for string-based coordinates
            for coord_name in ('location', 'realization'):
                if coord_name in ds.coords:
                    coord_data = ds[coord_name]
                    if coord_data.dtype.kind == 'S':  # Byte string
                        # Decode to Unicode strings
                        decoded_values = [val.decode('utf-8') if isinstance(val, bytes) else str(val) 
                                        for val in coord_data.values]
                        ds[coord_name] = (coord_data.dims, decoded_values)
            
            # DEBUG: Check slope coordinates after loading
            if 'slope_inclination' in ds.coords:
                logger.info(f"DEBUG read NetCDF: slope_inclination values AFTER load = {ds.slope_inclination.values}")
                logger.info(f"DEBUG read NetCDF: slope_inclination dims = {ds.slope_inclination.dims}")
            
            return xsnowDataset(ds)
        except Exception as e:
            logger.warning(f"Failed to read NetCDF file {source}: {e}")
            return None
    
    # Continue with normal .pro/.smet file processing
    try:
        pro_files, smet_files = _resolve_filepaths(source, recursive)
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
        lazy = total_files > 10
        logger.debug(f"Auto-detected lazy={lazy} for {total_files} files")

    # --- Step 4: Read and process data files ---
    # TODO: The datetime_start *_end parsing results in nan for inclination and azimuth!
    datetime_start, datetime_end = (time_sel.sel.start, time_sel.sel.stop) if time_sel and isinstance(time_sel.sel, slice) else (None, None)
    
    if lazy:
        chunks = chunks or {'time': 100, 'layer': -1}

        parser_kwargs['use_dask'] = lazy      # ← Set the flag
        parser_kwargs['chunks'] = chunks      # ← Set the chunks

        logger.info("Using lazy loading with Dask")
        pro_datasets = _read_pro_files_lazy(pro_files, datetime_start, datetime_end, n_cpus_use, logger, parser_kwargs)
    else:
        # logger.info("Using eager loading with multiprocessing")
        pro_datasets = _read_pro_files_eager(pro_files, datetime_start, datetime_end, n_cpus_use, logger, parser_kwargs)
    
    smet_datasets = _read_smet_files(smet_files, datetime_start, datetime_end, logger)

    # --- Step 5: Merge and finalize datasets ---
    all_datasets = pro_datasets + smet_datasets
    return _merge_and_finalize(all_datasets, time_sel, location_sel, slope_sel, realization_sel)


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
        logger.debug(f"Pre-scanning {len(pro_files)} .pro files to determine max layer dimension...")
        
        if n_cpus_use > 1 and len(pro_files) > 1:
            # Parallel pre-scan
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus_use, mp_context=ctx) as executor:
                layer_counts = list(executor.map(
                    _scan_pro_for_max_layers,
                    pro_files,
                    [add_sh] * len(pro_files),
                    [remove_soil] * len(pro_files)
                ))
        else:
            # Sequential pre-scan
            layer_counts = [_scan_pro_for_max_layers(fp, add_sh, remove_soil) for fp in pro_files]
        
        max_layers_to_use = max(layer_counts) if layer_counts else 200
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
        if raw_data and raw_data[1]:
            metadata, profiles = raw_data
            return _create_pro_dataset(metadata, profiles, max_layers, add_sh)
    except Exception as e:
        # Can't use logger in worker, just return None
        logging.warning(f"Warning: Failed to process {filepath.name}: {e}")
    
    return None



# --------------------------------------------------------------------------- #
#                      LAZY .PRO FILE PROCESSING PIPELINE                        #
# --------------------------------------------------------------------------- #

def _read_pro_files_lazy(
    pro_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    n_cpus_use: int,
    logger: logging.Logger,
    parser_kwargs: Dict
) -> List[xr.Dataset]:
    """
    Parses and creates datasets from .pro files with optional lazy loading.
    
    Behavior depends on the 'use_dask' flag in parser_kwargs:
    - If use_dask=True: Returns datasets backed by dask arrays (truly lazy).
      Data is not computed until user explicitly calls .compute() or .load().
      Suitable for datasets larger than RAM.
    - If use_dask=False: Uses Dask for parallel file processing, then calls
      compute() to create numpy-backed datasets. Faster for small datasets
      that fit in memory.

    Args:
        pro_files: A list of file paths to .pro files.
        datetime_start: An optional start timestamp for filtering.
        datetime_end: An optional end timestamp for filtering.
        n_cpus_use: The number of CPU cores to use for parallel processing.
        logger: A configured logger instance.
        parser_kwargs: Additional keyword arguments for the parser, including:
            - 'use_dask' (bool): Whether to create dask arrays (lazy) or numpy arrays (eager)
            - 'chunks' (dict): Chunk sizes if use_dask=True, e.g., {'time': 100, 'layer': -1}
            - 'max_layers': Max layers to pad to
            - 'remove_soil' (bool): Remove soil layers
            - 'add_surface_sh_as_layer' (bool): Add surface hoar as layer

    Returns:
        A list of xarray.Dataset objects. If use_dask=True, these contain
        dask arrays and are truly lazy. If use_dask=False, these contain
        numpy arrays and data is already loaded in memory.
    """
    if not pro_files:
        return []

    # Extract use_dask flag - default to True since this function is meant to be lazy
    use_dask = parser_kwargs.get('use_dask', True)
    
    # --- Stage 1: Parallel Pre-scan for Max Layers ---
    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    
    # Get the add_sh and remove_soil arguments to pass to the scanner
    add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
    remove_soil = parser_kwargs.get('remove_soil', True)
    
    if max_layers_arg == 'auto':
        logger.debug(f"Pre-scanning {len(pro_files)} .pro files to determine max layer dimension...")
        use_parallel = n_cpus_use > 1 and len(pro_files) > 1
        layer_counts = []
        if use_parallel:
            # Use 'spawn' context to avoid fork() deprecation warning
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_cpus_use,
                mp_context=ctx  
            ) as executor:
                future_to_file = {
                    executor.submit(_scan_pro_for_max_layers, fp, add_sh, remove_soil): fp for fp in pro_files
                }
                for future in concurrent.futures.as_completed(future_to_file):
                    layer_counts.append(future.result())
        else:
            layer_counts = [_scan_pro_for_max_layers(fp, add_sh, remove_soil) for fp in pro_files]
        
        max_layers_to_use = max(layer_counts) if layer_counts else 200
        logger.debug(f"Pre-scan complete. Max layers found: {max_layers_to_use}")
    else:
        max_layers_to_use = int(max_layers_arg)
    
    # --- Stage 2: Create Datasets ---
    if use_dask:
        # TRUE LAZY MODE: Create datasets with dask arrays (no compute!)
        logger.info("Creating lazy datasets backed by dask arrays...")
        logger.info("Data will NOT be computed until explicitly requested by user")
        
        final_datasets = []
        chunks = parser_kwargs.get('chunks', {'time': 100, 'layer': -1})
        
        for fp in pro_files:
            try:
                # Parse raw data (metadata and structure - small, can be eager)
                metadata, profiles = _read_pro_raw(
                    fp,
                    datetime_start=datetime_start,
                    datetime_end=datetime_end,
                    code_map=CODE_TO_NAME_MAP,
                    logger=logger,
                    remove_soil=remove_soil,
                    add_surface_sh_as_layer=add_sh
                )
                
                if profiles:
                    # Create dataset with dask arrays (lazy!)
                    ds = _create_pro_dataset(
                        metadata, 
                        profiles, 
                        max_layers_to_use, 
                        add_sh,
                        use_dask=True,  # Create dask arrays
                        chunks=chunks
                    )
                    if ds is not None:
                        final_datasets.append(ds)
                        
            except Exception as e:
                logger.warning(f"Failed to process {fp.name}: {e}")
        
        logger.info(f"Created {len(final_datasets)} lazy datasets (data NOT yet loaded into memory)")
        return final_datasets
        
    else:
        # PARALLEL PROCESSING MODE: Use Dask for parallelization, compute immediately
        logger.info("Building Dask graph for parallel dataset creation...")
        lazy_datasets = [
            _create_dataset_from_pro_file(
                fp, max_layers_to_use, datetime_start, datetime_end, logger, parser_kwargs
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

        # Step 2: Create the dataset if data was found
        if raw_data and raw_data[1]:  # Check if profiles list is not empty
            metadata, profiles = raw_data
            return _create_pro_dataset(metadata, profiles, max_layers, add_sh)
            
    except Exception as e:
        logger.error(f"Failed to create dataset from {filepath.name}", exc_info=e)
    
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



def _handle_duplicate_names(names: List[str]) -> List[str]:
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
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Duplicate station name '{name}' found. "
                    f"Renaming all occurrences with indices (e.g., {name}_0, {name}_1, ...)."
                )
                warned.add(name)
        else:
            unique_name = name
        
        unique_names.append(unique_name)
    
    return unique_names

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
    if logger is None:
        logger = logging.getLogger(__name__)

    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return {}, []

    code_map_to_use = code_map if code_map is not None else CODE_TO_NAME_MAP
    metadata: Dict[str, Any] = {}
    temp_profiles: List[Dict] = []
    
    in_header, in_data = False, False

    start_ts = pd.to_datetime(datetime_start, format='mixed') if datetime_start else None
    end_ts = pd.to_datetime(datetime_end, format='mixed') if datetime_end else None

    # --- HELPER FUNCTIONS ---
    def _process_and_append_profile(profile_data: Dict):
        """Helper to process a completed profile dictionary."""
        if not profile_data or 'timestamp' not in profile_data:
            return

        profile_ts = pd.to_datetime(profile_data['timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        if pd.notna(profile_ts):
            in_range = True
            if start_ts and profile_ts < start_ts:
                in_range = False
            if end_ts and profile_ts > end_ts:
                in_range = False
            
            if in_range:
                # Check for soil layers and slice the data if necessary
                if remove_soil and 'grain_type' in profile_data:
                    grain_types = profile_data['grain_type']
                    # Find the first occurrence of a soil layer
                    soil_indices = np.where((grain_types == 81) | (grain_types == 82))[0]
                    
                    if soil_indices.size > 0:
                        first_soil_idx = soil_indices[0]
                        # Create a new dictionary, slicing all array variables
                        # to keep only the elements BEFORE the first soil layer.
                        new_profile_data = {}
                        for key, val in profile_data.items():
                            if isinstance(val, np.ndarray) and val.size == grain_types.size:
                                new_profile_data[key] = val[:first_soil_idx]
                            else:
                                new_profile_data[key] = val
                        profile_data = new_profile_data
                
                height_array = profile_data.get('height', np.array([]))
                # Ensure height_array is not empty after potential slicing
                if height_array.size == 0:
                    return

                has_snow = np.nanmax(height_array) > 0

                sh_data = profile_data.get('SH_at_surface', np.array([]))
                has_sh = add_surface_sh_as_layer and sh_data.size >= 3 and sh_data[0] > 0
                
                if has_snow or has_sh:
                    temp_profiles.append(profile_data)
                    
    def _parse_and_process_block(lines_block: List[str]):
        """Parses a block of lines for one timestamp into a dict and processes it."""
        if not lines_block:
            return
        
        profile_dict = {}
        # The first line is always the timestamp
        ts_line = lines_block[0]
        profile_dict['timestamp'] = ts_line.split(',', 1)[1]
        
        # Process the rest of the lines for this block
        for profile_line in lines_block[1:]:
            try:
                param_code, *data_parts = profile_line.split(',')
                if (param_name := code_map_to_use.get(param_code)):
                    data_values = data_parts[1:] # Skip the layer count
                    profile_dict[param_name] = np.array(data_values, dtype=float)
            except (ValueError, IndexError):
                logger.warning(
                    f"Could not parse data in {file_path.name} at timestamp "
                    f"{profile_dict.get('timestamp', 'unknown')}. Line: '{profile_line}'"
                )
        _process_and_append_profile(profile_dict)

    # --- MAIN PARSING LOGIC ---
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_profile_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line == '[STATION_PARAMETERS]':
            in_header, in_data = True, False
        elif line == '[DATA]':
            in_header, in_data = False, True
        elif line.startswith('['):
            in_header, in_data = False, False
        elif in_header:
            for key, value in HEADER_MAP.items():
                if line.startswith(key):
                    metadata[value] = line.split('=', 1)[1].strip()
        elif in_data:
            # Detect the start of a new profile block
            if line.startswith("0500,"):
                # If we have lines from a previous block, process them now
                _parse_and_process_block(current_profile_lines)
                # Start a new block
                current_profile_lines = [line]
            else:
                # Add the current line to the block
                current_profile_lines.append(line)

    # After the loop, process the very last profile block in the file
    _parse_and_process_block(current_profile_lines)

    # POST-PROCESSING: Sanitize station name
    if 'station_name' not in metadata or not metadata['station_name']:
        # Use filename as fallback
        metadata['station_name'] = file_path.stem
        if logger:
            logger.warning(
                f"No StationName found in {file_path.name}, using filename: {file_path.stem}"
            )
    
    # Sanitize the station name
    metadata['station_name'] = _sanitize_station_name(metadata['station_name'])

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
    chunks: Dict[str, int] | None = None
) -> Optional[xr.Dataset]:
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

    The final dataset will have dimensions:
    ('location', 'time', 'slope', 'realization', 'layer')

    Args:
        metadata (Dict): A dictionary of station parameters from the .pro
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

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the structured
        snow profile data, or None if the input profiles list is empty.
    """
    if not profiles:
        return None

    chunks = chunks or {'time': 100, 'layer': -1}
        
    timestamps = pd.to_datetime([p['timestamp'] for p in profiles], format='%d.%m.%Y %H:%M:%S')
    
    all_params = sorted({
        key for p in profiles for key in p 
        if key not in ['timestamp', 'SH_at_surface']
    })

    data_vars = {}
    
    for param in all_params:
        shape = (1, len(profiles), 1, 1, max_layers)
        
        # Determine chunks
        chunk_spec = None
        if use_dask:
            chunk_spec = (1, chunks['time'], 1, 1, chunks['layer'])
        
        # Single line - same for both modes!
        arr = _create_array(shape, np.nan, np.float32, use_dask, chunk_spec)
        
        data_vars[param] = (("location", "time", "slope", "realization", "layer"), arr)

    for i, profile in enumerate(profiles):
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

        for param in all_params:
            param_data = profile.get(param, [])
            final_values = list(param_data[:num_snow_layers])
            
            while len(final_values) < num_snow_layers:
                final_values.append(np.nan)

            if has_sh:
                if param == 'grain_type':
                    final_values.append(660)
                elif param == 'grain_size':
                    final_values.append(sh_gsize)
                elif param == 'density':
                    final_values.append(sh_density)
                elif param == 'height':
                    last_height = final_values[-1] if final_values and not np.isnan(final_values[-1]) else 0.0
                    final_values.append(last_height + 0.001)
                else:
                    final_values.append(np.nan)

            if final_values:
                num_to_copy = min(len(final_values), max_layers)
                data_vars[param][1][0, i, 0, 0, :num_to_copy] = np.array(final_values)[:num_to_copy]

    # USE STRING STATION NAME DIRECTLY (not 'stationName')
    station_name = metadata.get('station_name', metadata.get('stationName', 'unknown'))
    
    slope_index = [0]
    ds = xr.Dataset(
        data_vars,
        coords={
            'location': [station_name],  # ✨ STRING COORDINATE!
            'time': timestamps,
            'slope': slope_index,
            'realization': [0],
            'layer': np.arange(max_layers),
            'latitude': ('location', [float(metadata.get('latitude', np.nan))]),
            'longitude': ('location', [float(metadata.get('longitude', np.nan))]),
            'altitude': ('location', [float(metadata.get('altitude', np.nan))]),
            'slope_inclination': ('slope', [
                int(round(float(metadata.get('slopeAngle', np.nan))))
                if not np.isnan(float(metadata.get('slopeAngle', np.nan)))
                else np.nan
            ]),
            'slope_azimuth': ('slope', [
                int(round(float(metadata.get('slopeAzi', np.nan))))
                if not np.isnan(float(metadata.get('slopeAzi', np.nan)))
                else np.nan
            ])
        }
    )

    ds.attrs = {
        'Conventions': 'CF-1.8',
        'title': f"Snowpack Profile for {station_name}",
        'source': 'SNOWPACK model output (.pro file)',
        'history': f'{pd.Timestamp.now(tz="UTC").isoformat()}: Created by xsnow.',
        'crs': 'EPSG:4326'
    }
    
    # Add variable metadata
    for var_name_hashable in list(ds.variables):
        var_name = str(var_name_hashable)
        if reg_entry := registry.get(var_name):
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', 'unknown')
            ds[var_name].attrs['units'] = reg_entry.get('si_units', 'unknown')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', var_name).replace('_', ' ')

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


def _scan_pro_for_max_layers(filepath: Path, add_sh: bool, remove_soil: bool = True) -> int:
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
    if logger is None:
        logger = logging.getLogger(__name__)

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
            ds['time'].attrs['timezone'] = int(tz_val)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timezone '{tz_val}' from file {filepath}")
            
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
            'latitude': ('location', [float(header.get('latitude', np.nan))]),
            'longitude': ('location', [float(header.get('longitude', np.nan))]),
            'altitude': ('location', [float(header.get('altitude', np.nan))]),
        })
    
    # FIXED: Try to get slope values from SMET header before defaulting to NaN
    # This prevents unnecessary NaN assignments when slope metadata is available
    if 'slope_inclination' not in ds.coords:
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
            'slope_inclination': ('slope', [slope_incl]),
            'slope_azimuth': ('slope', [slope_azi])
        })

    return ds

def _resolve_filepaths(source: Union[str, Path, Sequence[Union[str, Path]]],
                       recursive: bool = False
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
    time: Optional[TimeSelector],
    location: Optional[LocationSelector],
    slope: Optional[SlopeSelector],
    realization: Optional[RealizationSelector]
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

    Returns:
        A unified and filtered xsnowDataset, or None if no datasets were provided.
    """
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
    unique_names = _handle_duplicate_names(station_names)
    
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
                # Sort by slope_inclination if it exists, otherwise use 0 as default
                def get_slope_key(ds):
                    if 'slope_inclination' in ds.coords:
                        return ds.coords['slope_inclination'].values[0]
                    elif 'slope_inclination' in ds.data_vars:
                        return ds['slope_inclination'].values[0]
                    else:
                        return 0.0  # Default value if slope_inclination doesn't exist
                
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
                
                # Concatenate along slope dimension
                combined_profiles_ds = xr.concat(profile_dsets_cleaned, dim='slope', join='outer', coords='all')
                combined_profiles_ds['slope'] = np.arange(combined_profiles_ds.sizes['slope'])
                
                # Restore location-level coordinates (without slope dimension)
                combined_profiles_ds = combined_profiles_ds.assign_coords(location_coords)
            else:
                combined_profiles_ds = profile_dsets[0]

        # IMPORTANT: Profile datasets (from .pro files) must come FIRST in the merge list.
        # xr.merge with compat='override' keeps the FIRST dataset's coordinate values when there are conflicts.
        # This ensures valid slope_inclination/azimuth values from .pro files take precedence
        # over potentially NaN values from .smet files, preventing coordinate corruption.
        all_to_merge = ([combined_profiles_ds] if combined_profiles_ds is not None else []) + timeseries_dsets
        
        if not all_to_merge: 
            continue

        if len(all_to_merge) > 1:
            merged_loc_ds = xr.merge(all_to_merge, compat='override', join='outer')
            merged_datasets.append(merged_loc_ds)
        else:
            merged_datasets.append(all_to_merge[0])

    if not merged_datasets: 
        return None

    # Sort by station name for consistent ordering
    merged_datasets.sort(key=lambda ds: str(ds.location.values[0]))
    
    final_xr_dataset = xr.concat(
        merged_datasets,
        dim='location', 
        join='outer', 
        coords='all', 
        data_vars='all', 
        compat='override'
    )
    
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

    final_dataset = xsnowDataset(final_xr_dataset)
    
    # DEBUG: Check slope coordinates after merge
    if 'slope_inclination' in final_dataset.data.coords:
        logger.info(f"DEBUG _merge_and_finalize: slope_inclination values = {final_dataset.data.slope_inclination.values}")
        logger.info(f"DEBUG _merge_and_finalize: slope_inclination dims = {final_dataset.data.slope_inclination.dims}")
    
    return final_dataset.compute_z_depth() if 'height' in final_dataset.data else final_dataset


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
def update(existing_ds: xsnowDataset, source: Union[str, List[str], Path]) -> xsnowDataset:
    """
    Updates an existing xsnowDataset with new data from a source.

    This method is designed for workflows where you load a cached dataset
    (e.g., from NetCDF) and want to append new data from recent .pro or
    .smet files. It finds the last timestamp in the current data and only
    reads new data from the source that is more recent.
    
    Note: This method is optimized for adding new timestamps to existing data.
    If the new data has different dimensions (e.g., different slopes), the result
    may differ from loading all data together. For guaranteed consistency across
    all dimensions, use a full reload instead.

    Args:
        existing_ds (xsnowDataset): The dataset instance to be updated.
        source: Path to a file, a directory, or a list of files
            containing the new data.

    Returns:
        A new xsnowDataset instance with the combined data.
    """
    try:
        pro_files, smet_files = _resolve_filepaths(source)
        if not pro_files and not smet_files:
            logger.warning("Update source contains no data files.")
            return existing_ds
        resolved_source = pro_files + smet_files
    except (FileNotFoundError, TypeError) as e:
        logger.error(e)
        return existing_ds

    if existing_ds.data is None or existing_ds.data.time.size == 0:
        return read(resolved_source) or existing_ds
    
    last_timestamp = pd.to_datetime(existing_ds.data.time.max().values).isoformat()
    
    time_filter = TimeSelector(sel=slice(last_timestamp, None))
    new_data_obj = read(resolved_source, time=time_filter)

    if new_data_obj is None or new_data_obj.data.time.size == 0:
        return existing_ds

    # ========================================================================
    # FIX: Handle string-based location coordinates properly
    # ========================================================================
    
    # Get location names from both datasets
    def _get_location_names(ds):
        """Extract location names as strings, handling bytes if needed."""
        return [loc.decode('utf-8') if isinstance(loc, bytes) else str(loc) 
                for loc in ds.location.values]
    
    existing_locations = _get_location_names(existing_ds.data)
    new_locations = _get_location_names(new_data_obj.data)
    
    # Find common and unique locations
    common_locations = set(existing_locations) & set(new_locations)
    only_in_existing = set(existing_locations) - set(new_locations)
    only_in_new = set(new_locations) - set(existing_locations)
    
    logger.info(f"Update: {len(common_locations)} common, {len(only_in_new)} new locations")
    
    # Process based on whether we have common locations
    if common_locations:
        # Merge data for common locations along time dimension
        merged_parts = []
        
        # 1. Handle locations that exist in both datasets
        for loc_name in common_locations:
            # Select data for this location from both datasets
            existing_loc_data = existing_ds.data.sel(location=loc_name)
            new_loc_data = new_data_obj.data.sel(location=loc_name)
            
            # Align dimensions (except time)
            for dim in set(existing_loc_data.dims) | set(new_loc_data.dims):
                if dim == 'time':
                    continue
                
                existing_size = existing_loc_data.sizes.get(dim, 0)
                new_size = new_loc_data.sizes.get(dim, 0)
                
                if existing_size != new_size and existing_size > 0 and new_size > 0:
                    logger.warning(
                        f"Location '{loc_name}' dimension mismatch in '{dim}': "
                        f"existing={existing_size}, new={new_size}. Reindexing to match."
                    )
                    max_size = max(existing_size, new_size)
                    
                    if existing_size < max_size:
                        existing_loc_data = existing_loc_data.reindex(
                            {dim: range(max_size)}, fill_value=np.nan
                        )
                    if new_size < max_size:
                        new_loc_data = new_loc_data.reindex(
                            {dim: range(max_size)}, fill_value=np.nan
                        )
            
            # BEFORE concat: Save coordinates from existing dataset
            slope_coords_to_preserve = {}
            for coord_name in ['slope_inclination', 'slope_azimuth']:
                if coord_name in existing_loc_data.coords:
                    slope_coords_to_preserve[coord_name] = existing_loc_data[coord_name]
            
            # Concatenate along time dimension for this location
            merged_loc = xr.concat(
                [existing_loc_data, new_loc_data],
                dim='time',
                coords='all',
                compat='override'
            )
            
            # AFTER concat: Restore preserved coordinates
            if slope_coords_to_preserve:
                merged_loc = merged_loc.assign_coords(slope_coords_to_preserve)
            
            # Ensure location is preserved as a dimension
            if 'location' not in merged_loc.dims:
                # Drop location coordinate if it exists to avoid conflict
                if 'location' in merged_loc.coords:
                    merged_loc = merged_loc.drop_vars('location')  # ✅ Drop first
                # Add location as a new dimension with the correct value
                merged_loc = merged_loc.expand_dims(location=[loc_name])  # ✅ Then add

            merged_parts.append(merged_loc)  
                  
        # 2. Add locations that only exist in existing dataset
        for loc_name in only_in_existing:
            loc_data = existing_ds.data.sel(location=loc_name)
            if 'location' not in loc_data.dims:
                loc_data = loc_data.expand_dims(location=[loc_name])
            merged_parts.append(loc_data)
        
        # 3. Add locations that only exist in new dataset
        for loc_name in only_in_new:
            loc_data = new_data_obj.data.sel(location=loc_name)
            if 'location' not in loc_data.dims:
                loc_data = loc_data.expand_dims(location=[loc_name])
            merged_parts.append(loc_data)
        
        # Combine all location data
        combined = xr.concat(
            merged_parts,
            dim='location',
            coords='all',
            compat='override'
        )
        
    else:
        # No common locations - simple concatenation along location dimension
        # Handle dimension mismatches
        all_dims = set(existing_ds.data.dims) | set(new_data_obj.data.dims)
        
        for dim in all_dims:
            if dim in ('time', 'location'):
                continue
                
            existing_size = existing_ds.data.sizes.get(dim, 0)
            new_size = new_data_obj.data.sizes.get(dim, 0)
            
            if existing_size != new_size and existing_size > 0 and new_size > 0:
                logger.warning(
                    f"Dimension mismatch in '{dim}': existing={existing_size}, new={new_size}. "
                    f"Incremental update may differ from full reload."
                )
                max_size = max(existing_size, new_size)
                
                if existing_size < max_size:
                    existing_ds.data = existing_ds.data.reindex(
                        {dim: range(max_size)}, fill_value=np.nan
                    )
                if new_size < max_size:
                    new_data_obj.data = new_data_obj.data.reindex(
                        {dim: range(max_size)}, fill_value=np.nan
                    )
        
        # Concatenate along location dimension (different stations)
        combined = xr.concat(
            [existing_ds.data, new_data_obj.data],
            dim='location',
            coords='all',
            compat='override'
        )
    
    # Remove duplicate times (keep first occurrence)
    _, index = np.unique(combined['time'], return_index=True)
    combined = combined.isel(time=index)

    return xsnowDataset(combined)
    
@register_xsnowDataset_method
def to_netcdf(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """
    Saves the dataset to a NetCDF file.

    This method provides a convenient wrapper around the underlying
    `xarray.Dataset.to_netcdf()` method for easy caching and interoperability.
    
    The location mapping dictionary is converted to NetCDF-compatible format
    by encoding it as JSON in a string attribute.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .nc file.
        **kwargs: Additional keyword arguments passed to `xarray.Dataset.to_netcdf()`.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    
    # Create a copy to avoid modifying the original
    ds_to_save = ds.to_xarray().copy()
    
    # DEBUG: Check slope coordinates before saving
    if 'slope_inclination' in ds_to_save.coords:
        logger.info(f"DEBUG to_netcdf: slope_inclination values BEFORE save = {ds_to_save.slope_inclination.values}")
        logger.info(f"DEBUG to_netcdf: slope_inclination dims = {ds_to_save.slope_inclination.dims}")
    
    # ENCODE STRING COORDINATES FOR NETCDF
    for coord_name in ("location", "realization"):
        if coord_name in ds_to_save.coords:
            coord_data = ds_to_save[coord_name]
            
            # Check if coordinate is string-like
            if coord_data.dtype.kind in ("U", "S", "O"):
                # Determine max string length
                if coord_data.dtype.kind == "U":
                    max_len = max(len(str(x)) for x in coord_data.values)
                elif coord_data.dtype.kind == "S":
                    max_len = coord_data.dtype.itemsize
                else:  # Object dtype
                    max_len = max(len(str(x)) for x in coord_data.values)
                
                # Set encoding to fixed-length string
                ds_to_save[coord_name].encoding = {
                    "dtype": f"S{max_len}",
                    "compression": "zlib",
                    "complevel": 4
                }
                
                logger.debug(f"Encoding {coord_name} as S{max_len}")
    ds_to_save.to_netcdf(path, **kwargs)
    logging.info(f"Dataset successfully saved to {path}")

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
        if num_locations > max_files:
            raise ValueError(
                f"Dataset contains {num_locations} locations, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )
            
    if ds.sizes.get('location') > 1:
        raise ValueError("to_smet() only supports datasets with a single location.")

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

    if ds.sizes.get('location') > 1:
        raise ValueError("to_pro() only supports datasets with a single location.")

    # Squeeze location but preserve time dimension
    single_loc_ds = ds.data.squeeze(dim='location', drop=False)
    
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
            slope_val = int(round(single_loc_ds.slope_inclination.values.item()))
        except (AttributeError, ValueError):
            slope_val = -999
        f.write(f"SlopeAngle= {slope_val:.0f}\n")
        
        # Handle slope azimuth
        try:
            azimuth = int(round(single_loc_ds.slope_azimuth.values.item()))
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