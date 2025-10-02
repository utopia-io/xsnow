# xsnow_io.py
"""
Handles all data parsing and input/output operations for the xsnow package.

This module is the primary engine for reading and writing data in various
formats (e.g., SNOWPACK .pro, .smet, NetCDF).
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union, Tuple, Sequence
import concurrent.futures

import numpy as np
import pandas as pd
import xarray as xr

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
    'Altitude=': 'altitude',
    'Latitude=': 'latitude',
    'Longitude=': 'longitude',
    'SlopeAngle=': 'slopeAngle',
    'SlopeAzi=': 'slopeAzi',
    'StationName=': 'stationName'
}

def _parse_pro_worker(
    filepath: Path,
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    code_map: Dict,
    logger: logging.Logger,
    remove_soil: bool
) -> Optional[Tuple[Dict, List[Dict]]]:
    """
    Internal worker to parse a single .pro file into a raw data structure.

    This function is designed to be run in a separate process for parallel reading.
    It acts as a simple wrapper around the `_read_pro_raw` function, making it
    compatible with `concurrent.futures`.

    Args:
        filepath: Path to the .pro file.
        datetime_start: Start timestamp for filtering profiles.
        datetime_end: End timestamp for filtering profiles.
        code_map: Mapping from parameter codes to variable names.
        logger logging.Logger: A pre-configured
            logger instance. If None, the default module logger is used.

    Returns:
        A tuple containing the station metadata and a list of raw profile dicts. Or None if parsing fails
    """
    try:
        return _read_pro_raw(filepath, datetime_start, datetime_end, code_map=code_map, logger=logger, remove_soil=remove_soil)
    except Exception as e:
        logger.error(f"Failed to parse .pro file {filepath.name}", exc_info=e)
        return None

def _read_pro_raw(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    remove_soil: bool = True
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
    current_ts_data: Dict = {}
    in_header, in_data = False, False

    start_ts = pd.to_datetime(datetime_start) if datetime_start else None
    end_ts = pd.to_datetime(datetime_end) if datetime_end else None

    def test_timestamp(timestamp: str) -> bool:
        if not start_ts and not end_ts:
            return True
        profile_ts = pd.to_datetime(current_ts_data['timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        if not pd.notna(profile_ts):
            return False
        if start_ts and profile_ts < start_ts:
            return False
        if end_ts and profile_ts > end_ts:
            return False
        return True

    def add_data():
        if not current_ts_data:
            return
        if not test_timestamp(current_ts_data['timestamp']):
            return
        # --- Soil Removal Logic ---
        if remove_soil and 'grain_type' in current_ts_data:
            grain_types = current_ts_data['grain_type']
            is_soil = (grain_types == 81) | (grain_types == 82)
            soil_indices = np.where(is_soil)[0]

            if soil_indices.size > 0:
                first_soil_idx = soil_indices[0]
                for key, arr in current_ts_data.items():
                    if isinstance(arr, np.ndarray) and arr.ndim == 1:
                        current_ts_data[key] = arr[:first_soil_idx]

        # Only append if there are still layers left after slicing
        if current_ts_data.get('height', np.array([])).size > 0:
            temp_profiles.append(current_ts_data)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
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
                if line.startswith("0500,"):
                    add_data()
                    current_ts_data = {'timestamp': line[5:]}
                else:
                    if (param_name := code_map_to_use.get(line[:line.index(',')])):
                        try:
                            current_ts_data[param_name] = np.fromstring(line, dtype=float, sep=',')[2:]
                        except ValueError:
                            ts = current_ts_data.get('timestamp', 'unknown timestamp')
                            logger.warning(
                                f"Could not parse data for param {param_name} in {Path(filepath).name} "
                                f"at timestamp {ts} (line approx {i+1}). Skipping parameter."
                            )

    # This block handles the very last profile in the file
    add_data()
    return metadata, temp_profiles


def _create_pro_dataset_worker(
    raw_data: Tuple[Dict, List[Dict]],
    max_layers: int,
    logger: logging.Logger
) -> Optional[xr.Dataset]:
    """
    Internal worker to convert raw parsed .pro data into a padded xarray.Dataset.

    This function is designed to be run in a separate process for parallel
    dataset creation. It takes the raw output from `_parse_pro_worker` and
    transforms it into a fully-formed, padded `xarray.Dataset`.

    Args:
        raw_data: A tuple of metadata and raw profile data.
        max_layers: The number of layers to pad the dataset to.
        logger logging.Logger: A pre-configured
            logger instance. If None, the default module logger is used.

    Returns:
        An xarray.Dataset for a single station, or None if no profiles were found or raw_data is None.
    """
    if raw_data is None:
        return None
    try:
        metadata, profiles = raw_data
        if not profiles:
            return None
        return _create_pro_dataset(metadata, profiles, max_layers)
    except Exception as e:
        station_name = raw_data[0].get('stationName', 'unknown')
        logger.error(f"Failed to create dataset for station '{station_name}'", exc_info=e)
        return None


# xsnow_io.py

def _resolve_filepaths(source: Union[str, Path, Sequence[Union[str, Path]]]) -> Tuple[List[Path], List[Path]]:
    """
    Resolves the source argument into separate lists of .pro and .smet files.

    Args:
        source: The data source, which can be a directory path, a single
            file path, or a list of paths (including a mix of files and directories).

    Returns:
        A tuple containing two lists: one for .pro file paths and one for
        .smet file paths.
        
    Raises:
        TypeError: If the source is of an unsupported type.
    """
    paths_to_process = []
    if isinstance(source, (str, Path)):
        # If the source is a single item, wrap it in a list to process uniformly
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
            # If it's a directory, glob for files and add them to our list
            resolved_files.extend(p.glob('*.pro'))
            resolved_files.extend(p.glob('*.smet'))
        else:
            # If it's a file, just add it
            resolved_files.append(p)

    # Use dict.fromkeys to get unique files while preserving order, then sort
    unique_files = sorted(list(dict.fromkeys(resolved_files)))

    pro_files = [f for f in unique_files if f.suffix == '.pro']
    smet_files = [f for f in unique_files if f.suffix == '.smet']
    
    return pro_files, smet_files

#---------------------------------------------------------------
# Read() function helper functions
#---------------------------------------------------------------
def _read_pro_files(
    pro_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    n_cpus_use: int,
    logger: logging.Logger,
    parser_kwargs: Dict
) -> List[xr.Dataset]:
    """
    Parses and creates datasets from a list of .pro files.

    This function orchestrates the parallel reading of raw data and the
    subsequent parallel creation of xarray.Dataset objects.

    Args:
        pro_files: A list of file paths to .pro files.
        datetime_start: An optional start timestamp for filtering.
        datetime_end: An optional end timestamp for filtering.
        n_cpus_use: The number of CPU cores to use for parallel processing.
        parser_kwargs: Additional keyword arguments for the parser,
            such as `max_layers`.

    Returns:
        A list of xarray.Dataset objects, one for each successfully
        processed .pro file.
    """
    if not pro_files:
        return []

    # Extract remove_soil arg with a default of True
    remove_soil = parser_kwargs.get('remove_soil', True)

    # Step 1: Parallel parsing
    raw_pro_data = []
    use_parallel = n_cpus_use > 1 and len(pro_files) > 1
    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus_use) as executor:
            futures = {executor.submit(_parse_pro_worker, fp, datetime_start, datetime_end, CODE_TO_NAME_MAP, logger, remove_soil) for fp in pro_files}
            for future in concurrent.futures.as_completed(futures):
                if result := future.result():
                    raw_pro_data.append(result)
    else:
        for fp in pro_files:
            if result := _parse_pro_worker(fp, datetime_start, datetime_end, CODE_TO_NAME_MAP, logger, remove_soil):
                raw_pro_data.append(result)

    if not raw_pro_data:
        return []

    # Step 2: Determine max layers
    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    if max_layers_arg == 'auto':
        max_layers_to_use = max((len(p.get('height', [])) for _, profiles in raw_pro_data for p in profiles if profiles), default=200)
    else:
        max_layers_to_use = int(max_layers_arg)

    # Step 3: Parallel dataset creation
    pro_datasets = []
    valid_raw_data = [rd for rd in raw_pro_data if rd and rd[1]]
    if use_parallel and len(valid_raw_data) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus_use) as executor:
            futures = {executor.submit(_create_pro_dataset_worker, raw_data, max_layers_to_use, logger) for raw_data in valid_raw_data}
            for future in concurrent.futures.as_completed(futures):
                if ds := future.result():
                    pro_datasets.append(ds)
    else:
        for raw_data in valid_raw_data:
            if ds := _create_pro_dataset_worker(raw_data, max_layers_to_use, logger):
                pro_datasets.append(ds)
                
    return pro_datasets


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

def _merge_and_finalize(
    all_datasets: List[xr.Dataset],
    time: Optional[TimeSelector],
    location: Optional[LocationSelector],
    slope: Optional[SlopeSelector],
    realization: Optional[RealizationSelector]
) -> Optional[xsnowDataset]:
    """Merges, re-indexes, and applies final filters to the datasets."""
    if not all_datasets:
        return None

    # 1. Check for and enforce timezone consistency across all source files.
    final_timezone = None
    for ds in all_datasets:
        if 'time' in ds and 'timezone' in ds.time.attrs:
            current_tz = ds.time.attrs['timezone']
            if final_timezone is None:
                final_timezone = current_tz
            elif final_timezone != current_tz:
                raise ValueError(
                    f"Conflicting timezones found in source files: "
                    f"'{final_timezone}' and '{current_tz}'. "
                    "Please ensure all SMET files use the same timezone."
                )

    # Group datasets by their location (station name)
    grouped_by_location = {}
    for ds in all_datasets:
        loc = ds.location.item()
        grouped_by_location.setdefault(loc, []).append(ds)
        
    # Merge datasets for each location, enforcing coordinate consistency
    
    merged_datasets = []
    for dsets in grouped_by_location.values():    
        # 1. Separate datasets based on the presence of the 'layer' dimension.
        profile_dsets = [ds for ds in dsets if 'layer' in ds.dims]
        timeseries_dsets = [ds for ds in dsets if 'layer' not in ds.dims]

        combined_profiles_ds = None
        if profile_dsets:
            # 2. If there are profile datasets, concatenate them along the slope dimension
            if len(profile_dsets) > 1:
                # Sort the datasets by the slope_inclination coordinate before concatenating
                profile_dsets.sort(key=lambda ds: ds.slope_inclination.item())
                
                # Add coords='minimal' as recommended by the error for robustness
                combined_profiles_ds = xr.concat(profile_dsets, dim='slope', join='outer', coords='minimal')
                
                # Create a new, sequential integer index for the concatenated dimension
                new_slope_index = np.arange(combined_profiles_ds.sizes['slope'])
                combined_profiles_ds['slope'] = new_slope_index
            else:
                combined_profiles_ds = profile_dsets[0]

        # 3. Create a list of all datasets to be merged for this location
        all_to_merge = ([combined_profiles_ds] if combined_profiles_ds is not None else []) + timeseries_dsets
        
        if not all_to_merge:
            continue

        # 4. Merge the combined profile data with the time-series data
        if len(all_to_merge) > 1:
            # Merge with compat='override'. 
             merged_loc_ds = xr.merge(all_to_merge, compat='override', join='outer') # The value from the last dataset in the merge list will silently overwrite the values from the earlier ones
             merged_datasets.append(merged_loc_ds)
        else:
             merged_datasets.append(all_to_merge[0])
    if not merged_datasets:
        return None

    # Re-index location dimension from strings to integers
    station_names = sorted([ds.location.item() for ds in merged_datasets])
    name_to_int = {name: i for i, name in enumerate(station_names)}
    int_to_name = dict(enumerate(station_names))
    
    reindexed_datasets = []
    for ds in merged_datasets:
        station_name = ds.location.item()
        ds['location'] = [name_to_int[station_name]]
        reindexed_datasets.append(ds)

    # Concatenate all location datasets into the final dataset
    final_xr_dataset = xr.concat(
        sorted(reindexed_datasets, key=lambda ds: ds.location.item()),
        dim='location',
        join='outer',
        coords='minimal',
        data_vars='all',
        compat='override'
    )
    final_xr_dataset['location'].attrs['mapping'] = int_to_name

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
    return final_dataset.compute_z_depth() if 'height' in final_dataset.data else final_dataset


def read(
    source: Union[str, List[str], Path, List[Path]],
    time: Optional[TimeSelector] = None,
    location: Optional[LocationSelector] = None,
    slope: Optional[SlopeSelector] = None,
    realization: Optional[RealizationSelector] = None,
    n_cpus_use: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
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
        time: A `TimeSelector` object for filtering data by time.
        location: A `LocationSelector` object for filtering by station ID.
        slope: A `SlopeSelector` for filtering by slope angle.
        realization: A `RealizationSelector` for filtering by model realization.
        n_cpus_use (Optional[int]): The number of CPU cores to use. If not
            provided, it defaults to min(32, total_cores - 1), with a
            minimum of 1. Set to 1 for sequential processing.
        logger (Optional[logging.Logger]): An optional, pre-configured
            logger instance. If None, the default module logger is used.
        **parser_kwargs: Additional keyword arguments passed to the underlying
            parser functions, such as `max_layers='auto'` or
            `remove_soil=True`.

    Returns:
        A unified `xsnowDataset` object containing the data from all sources,
        or `None` if no valid data files are found.
    """
    # --- Step 1: Setup Logger and CPU count ---
    if logger is None:
        logger = logging.getLogger(__name__)

    if n_cpus_use is None:
        cpu_count = os.cpu_count()
        n_cpus_use = min(32, cpu_count - 1) if cpu_count and cpu_count > 1 else 1
    
    # --- Step 2: Resolve file paths ---
    if isinstance(source, (str, Path)) and Path(source).suffix == '.nc':
        try:
            return xsnowDataset(xr.open_dataset(source))
        except Exception as e:
            logger.warning(f"Failed to read NetCDF file {source}: {e}")
            return None
            
    try:
        pro_files, smet_files = _resolve_filepaths(source)
    except (FileNotFoundError, TypeError) as e:
        logger.error(e)
        return None

    if not pro_files and not smet_files:
        logger.warning("No data files found in the specified source.")
        return None

    # --- Step 3: Read and process data files ---
    datetime_start, datetime_end = (time.sel.start, time.sel.stop) if time and isinstance(time.sel, slice) else (None, None)
    
    pro_datasets = _read_pro_files(pro_files, datetime_start, datetime_end, n_cpus_use, logger, parser_kwargs)
    smet_datasets = _read_smet_files(smet_files, datetime_start, datetime_end, logger)

    # --- Step 4: Merge and finalize datasets ---
    all_datasets = pro_datasets + smet_datasets
    return _merge_and_finalize(all_datasets, time, location, slope, realization)


def _create_pro_dataset(
    metadata: Dict,
    profiles: List[Dict],
    max_layers: int
) -> Optional[xr.Dataset]:
    """
    Creates a padded, CF-compliant xarray.Dataset from raw parsed .pro data.

    This function transforms the intermediate raw data from `_read_pro_raw`
    into a fully-formed, 5-dimensional xarray.Dataset. It pads all snow
    profile arrays with np.nan up to `max_layers` to ensure consistent array
    shapes across all time steps. CF-compliant metadata is attached to the
    dataset and its variables.

    The final dataset will have dimensions:
    ('location', 'time', 'slope', 'realization', 'layer')

    Args:
        metadata: A dictionary of station parameters from the .pro file header.
        profiles: A list of dictionaries, where each represents a single profile
            at a specific timestamp.
        max_layers: The number of layers to pad the profile data to. This ensures
            all profile variables have the same shape along the 'layer' dimension.

    Returns:
        An xarray.Dataset containing the structured snow profile data,
        or None if the input profiles list is empty.
    """
    if not profiles:
        return None

    timestamps = pd.to_datetime([p['timestamp'] for p in profiles], format='%d.%m.%Y %H:%M:%S')
    all_params = sorted({key for p in profiles for key in p if key != 'timestamp'})

    data_vars = {}
    for param in all_params:
        shape = (1, len(profiles), 1, 1, max_layers)
        arr = np.full(shape, np.nan, dtype=np.float32)
        data_vars[param] = (("location", "time", "slope", "realization", "layer"), arr)

    for i, profile in enumerate(profiles):
        num_layers = len(profile.get('height', []))
        for param in all_params:
            if param in profile:
                values = profile.get(param)
                if values is not None:
                    layers_to_copy = min(num_layers, max_layers)
                    data_vars[param][1][0, i, 0, 0, :layers_to_copy] = np.array(values)[:layers_to_copy]

    station_name = metadata.get('stationName', 'unknown')

    slope_index = [0]
    ds = xr.Dataset(
        data_vars,
        coords={
            'location': [station_name],
            'time': timestamps,
            'slope': slope_index, # Integer index for the dimension
            'realization': [0],
            'layer': np.arange(max_layers),
            'latitude': ('location', [float(metadata.get('latitude', np.nan))]),
            'longitude': ('location', [float(metadata.get('longitude', np.nan))]),
            'altitude': ('location', [float(metadata.get('altitude', np.nan))]),
            'slope_inclination': ('slope', [float(metadata.get('slopeAngle', np.nan))]),
            'slope_azimuth': ('slope', [float(metadata.get('slopeAzi', np.nan))])
        }
    )

    ds.attrs = {
        'Conventions': 'CF-1.8',
        'title': f"Snowpack Profile for {station_name}",
        'source': 'SNOWPACK model output (.pro file)',
        'history': f'{pd.Timestamp.now(tz="UTC").isoformat()}: Created by xsnow.',
        'crs': 'EPSG:4326'
    }
    for var_name_hashable in list(ds.variables):
        var_name = str(var_name_hashable)
        if reg_entry := registry.get(var_name):
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', 'unknown')
            ds[var_name].attrs['units'] = reg_entry.get('si_units', 'unknown')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', var_name).replace('_', ' ')

    return ds

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
   # Use the provided logger or fall back to the module's default
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
                # Append height to name, e.g., 'VW' becomes 'VW_10m'
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
        df = df[df.index >= pd.to_datetime(datetime_start)]
    if datetime_end:
        df = df[df.index <= pd.to_datetime(datetime_end)]
        
    if df.empty:
        return None

    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename({'timestamp': 'time'})
    # Hnadle cases where a timezone (tz) from a SMET file header that doesn't contain a tz field.
    tz_val = header.get('tz')
    if tz_val is not None:
        try:
            ds['time'].tz.set_attr(int(tz_val))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timezone '{tz_val}' from file {filepath}")
    station_name = header.get('station_name', header.get('station_id', file_path.stem))
    
    # Expand dims to be consistent with the 5D structure from .pro files
    ds = ds.expand_dims(location=[station_name], slope=[0], realization=[0])
    
    # Assign coordinates for the new dimensions to ensure proper merging later
    ds = ds.assign_coords({
        'latitude': ('location', [float(header.get('latitude', np.nan))]),
        'longitude': ('location', [float(header.get('longitude', np.nan))]),
        'altitude': ('location', [float(header.get('altitude', np.nan))]),
        'slope_inclination': ('slope', [np.nan]), # Use NaN as SMET has no slope info
        'slope_azimuth': ('slope', [np.nan])      # Use NaN as SMET has no slope info
    })

    return ds

# ================================================================= #
#                           WRITE FUNCTIONS                         #
# ================================================================= #

@register_xsnowDataset_method
def update(existing_ds: xsnowDataset, source: Union[str, List[str], Path]) -> xsnowDataset:
    """
    Updates a dataset with new data from a source.

    This function is designed for workflows where you load a cached dataset
    (e.g., from NetCDF) and want to append new data from recent .pro or
    .smet files. It finds the last timestamp in the current data and only
    reads new data from the source that is more recent.

    Args:
        existing_ds (xsnowDataset): The dataset instance to be updated.
        source: Path to a file, a directory, or a list of files
            containing the new data.

    Returns:
        A new xsnowDataset instance with the combined data.
    """
    # Resolve filepaths first to exit early if no files are found
    try:
        pro_files, smet_files = _resolve_filepaths(source)
        if not pro_files and not smet_files:
            logger.warning("Update source contains no data files.")
            return existing_ds
        # Pass the resolved list of files to read()
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

    # Combine and deduplicate the datasets
    combined = xr.concat([existing_ds.data, new_data_obj.data], dim="time")
    _, index = np.unique(combined['time'], return_index=True)
    combined = combined.isel(time=index)

    return xsnowDataset(combined)

    
@register_xsnowDataset_method
def to_netcdf(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """
    Saves the dataset to a NetCDF file.

    This method provides a convenient wrapper around the underlying
    `xarray.Dataset.to_netcdf()` method for easy caching and interoperability.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .nc file.
        **kwargs: Additional keyword arguments passed to `xarray.Dataset.to_netcdf()`.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    ds.to_xarray().to_netcdf(path, **kwargs)
    print(f"Dataset successfully saved to {path}")

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
    
    # Check if the number of locations exceeds the specified maximum
    if max_files is not None:
        num_locations = ds.sizes.get('location')
        if num_locations > max_files:
            raise ValueError(
                f"Dataset contains {num_locations} locations, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )
            
    if ds.sizes.get('location') > 1:
        raise ValueError("to_smet() only supports datasets with a single location.")

    # Select variables that are time-series (do not have a 'layer' dimension)
    smet_vars = [name for name, var in ds.data.data_vars.items() if 'layer' not in var.dims]
    if not smet_vars:
        print("Warning: No time-series variables found to write to SMET file.")
        return

    df = ds.data[smet_vars].squeeze().to_dataframe()

    # Write header
    with open(path, 'w') as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        f.write(f"station_name = {ds.location.item()}\n")
        f.write(f"latitude = {ds.latitude.item():.4f}\n")
        f.write(f"longitude = {ds.longitude.item():.4f}\n")
        f.write(f"altitude = {ds.altitude.item():.0f}\n")
        f.write("nodata = -999\n")
        # SMET convention uses 'timestamp' as the first field
        fields = "timestamp " + " ".join(df.columns)
        f.write(f"fields = {fields}\n")
        f.write("[DATA]\n")

    # Append data, formatting timestamp correctly
    df.index.name = 'timestamp'
    df.to_csv(path, sep=' ', mode='a', header=False, date_format='%Y-%m-%dT%H:%M:%S')
    print(f"Dataset successfully saved to {path}")


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
        
    # Check if the number of profiles exceeds the specified maximum
    if max_files is not None:
        num_profiles = ds.sizes.get('time')
        if num_profiles > max_files:
            raise ValueError(
                f"Dataset contains {num_profiles} profiles, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )

    if ds.sizes.get('location') > 1:
        raise ValueError("to_pro() only supports datasets with a single location.")

    single_loc_ds = ds.data.squeeze()
    profile_vars = {name: var for name, var in single_loc_ds.data_vars.items() if 'layer' in var.dims}

    with open(path, 'w') as f:
        # Write Header
        f.write("[STATION_PARAMETERS]\n")
        f.write(f"StationName= {single_loc_ds.location.item()}\n")
        f.write(f"Latitude= {single_loc_ds.latitude.item():.4f}\n")
        f.write(f"Longitude= {single_loc_ds.longitude.item():.4f}\n")
        f.write(f"Altitude= {single_loc_ds.altitude.item():.0f}\n")
        f.write(f"SlopeAngle= {single_loc_ds.slope.item():.1f}\n")
        f.write(f"SlopeAzi= {getattr(single_loc_ds, 'slope_aspect', 0.0):.1f}\n")
        f.write("[DATA]\n")

        # Write Data for each timestamp
        for t in single_loc_ds.time:
            profile = single_loc_ds.sel(time=t)
            timestamp_str = pd.to_datetime(t.item()).strftime('%d.%m.%Y %H:%M:%S')
            f.write(f"0500,{timestamp_str}\n")
            
            for var_name, code in registry.items():
                if var_name in profile_vars:
                    # Get the data, drop NaNs, and format for writing
                    data_array = profile[var_name].dropna(dim='layer').values
                    if data_array.size > 0:
                        num_layers = data_array.size
                        data_str = ",".join(map(str, data_array.round(4)))
                        f.write(f"{code},{num_layers},{data_str}\n")
                        
    print(f"Dataset successfully saved to {path}")

@register_xsnowDataset_method
def to_json(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """
    Saves the dataset to a structured JSON file. (Skeleton)

    This function will export the complete dataset, including all dimensions,
    coordinates, variables, and attributes, into a JSON format.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .json file.
        **kwargs: Additional keyword arguments for JSON formatting.
    """
    raise NotImplementedError("to_json() is not yet implemented.")


@register_xsnowDataset_method
def to_caaml(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """
    Saves snow profile data to a CAAML V6.0 XML file. (Skeleton)

    This function will format and export snow profile observations into the
    standard CAAML (Canadian Avalanche Association Markup Language) format,
    which is widely used for avalanche safety data exchange.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path for the .xml file.
        **kwargs: Additional keyword arguments for CAAML formatting.
    """
    raise NotImplementedError("to_caaml() is not yet implemented.")


@register_xsnowDataset_method
def to_crocus(ds: xsnowDataset, path: Union[str, Path], **kwargs):
    """
    Saves snow profile data to a Crocus model input file. (Skeleton)

    This function will export a snow profile into the specific ASCII format
    required for initializing or forcing the Crocus snowpack model.

    Args:
        ds (xsnowDataset): The dataset instance to save.
        path (Union[str, Path]): The destination file path.
        **kwargs: Additional keyword arguments for Crocus formatting.
    """
    raise NotImplementedError("to_crocus() is not yet implemented.")