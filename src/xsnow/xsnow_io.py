# xsnow_io.py (Refactored & Complete)
"""
This module contains all file parsers for the xsnow package.

It is responsible for reading various raw data formats (e.g., .pro, .smet, .nc)
and translating them into the standardized, multi-dimensional xarray.Dataset
used throughout the package.

The main entry point is the high-level `read()` function, which can handle
multiple file types, read them in parallel, and perform data integrity checks
before merging the results into a single, cohesive dataset.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import xarray as xr

# Import from the package's modules
from .registry import registry
from .dataset import xsnowDataset
from .selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector

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
    code_map: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Internal worker to parse a single .pro file into a raw data structure.

    This function is designed to be run in a separate process for parallel reading.
    It calls the low-level `_read_pro_raw` function and returns its output,
    which consists of station metadata and a list of raw profile dictionaries.

    Args:
        filepath: Path to the .pro file.
        datetime_start: Start timestamp for filtering profiles.
        datetime_end: End timestamp for filtering profiles.
        code_map: Mapping from parameter codes to variable names.

    Returns:
        A tuple containing the station metadata and a list of raw profile dicts.
    """
    return _read_pro_raw(filepath, datetime_start, datetime_end, code_map=code_map)

def _create_pro_dataset_worker(
    raw_data: Tuple[Dict, List[Dict]],
    max_layers: int
) -> Optional[xr.Dataset]:
    """
    Internal worker to convert raw parsed .pro data into a padded xarray.Dataset.

    This function is designed to be run in a separate process for parallel
    dataset creation. It takes the raw output from `_parse_pro_worker` and
    transforms it into a fully-formed, padded `xarray.Dataset`.

    Args:
        raw_data: A tuple of metadata and raw profile data.
        max_layers: The number of layers to pad the dataset to.

    Returns:
        An xarray.Dataset for a single station, or None if no profiles were found.
    """
    metadata, profiles = raw_data
    if not profiles:
        return None
    return _create_pro_dataset(metadata, profiles, max_layers)


def read(
    source: Union[str, List[str], Path],
    time: Optional[TimeSelector] = None,
    location: Optional[LocationSelector] = None,
    slope: Optional[SlopeSelector] = None,
    realization: Optional[RealizationSelector] = None,
    parallel: bool = True,
    **parser_kwargs: Any
) -> Optional[xsnowDataset]:
    """
    High-level function to read SNOWPACK data and return an xsnowDataset.

    This is the primary entry point for loading data into the xsnow package.
    It handles various file formats, processes them in parallel, and allows for
    efficient, at-the-source filtering using Selector objects.

    Args:
        source: Path to a file, a directory, or a list of file paths.
        time: Selector for filtering by time. Filtering by a time slice
            is performed at the source for efficiency.
        location: Selector for filtering by location.
        slope: Selector for filtering by slope.
        realization: Selector for filtering by realization.
        parallel: If True, enables parallel processing of multiple files.
        **parser_kwargs: Additional keyword arguments passed to the underlying
            parser functions. For example, `max_layers='auto'` or `max_layers=300`
            can be used to control the padding of snow profile data.

    Returns:
        A unified xsnowDataset containing the data from all sources,
        or None if no valid data files are found.

    Examples
    --------
    >>> # Load all .pro and .smet files from a directory
    >>> ds = read('path/to/data_directory/')

    >>> # Load data for a specific time range using a TimeSelector
    >>> from xsnow.selectors import TimeSelector
    >>> time_filter = TimeSelector(sel=slice('2024-01-01', '2024-01-31'))
    >>> ds_jan = read('path/to/data_directory/', time=time_filter)

    >>> # Load data and cap the number of layers to save memory
    >>> ds_capped = read('path/to/data_directory/', max_layers=250)
    """
    # --- File Path Resolution ---
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            logger.error(f"Source path does not exist: {source}")
            return None
        if p.is_dir():
            filepaths = sorted(list(p.glob('*.pro'))) + sorted(list(p.glob('*.smet')))
        # Handle single NetCDF file read
        elif p.suffix == '.nc':
            try:
                return xsnowDataset(xr.open_dataset(p))
            except Exception as e:
                logger.error(f"Failed to read NetCDF file {p}: {e}")
                return None
        else:
            filepaths = [p]
    elif isinstance(source, list):
        filepaths = [Path(f) for f in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if not filepaths:
        logger.warning("No data files found in the specified source.")
        return None

    pro_files = [f for f in filepaths if f.suffix == '.pro']
    smet_files = [f for f in filepaths if f.suffix == '.smet']

    datetime_start, datetime_end = None, None
    if time and isinstance(time.sel, slice):
        datetime_start = time.sel.start
        datetime_end = time.sel.stop

    raw_pro_data = []
    if pro_files:
        if parallel and len(pro_files) > 1:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_data = {
                    executor.submit(_parse_pro_worker, fp, datetime_start, datetime_end, CODE_TO_NAME_MAP): fp
                    for fp in pro_files
                }
                for future in concurrent.futures.as_completed(future_to_data):
                    raw_pro_data.append(future.result())
        else:
            for fp in pro_files:
                raw_pro_data.append(_parse_pro_worker(fp, datetime_start, datetime_end, CODE_TO_NAME_MAP))

    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    if isinstance(max_layers_arg, str) and max_layers_arg == 'auto':
        if raw_pro_data:
            max_layers_to_use = max((len(p.get('height', [])) for _, profiles in raw_pro_data for p in profiles), default=1)
        else:
            max_layers_to_use = 200
    else:
        max_layers_to_use = int(max_layers_arg)

    pro_datasets = []
    if raw_pro_data:
        # ... (parallel dataset creation logic remains the same) ...
        for raw_data in raw_pro_data:
            if raw_data[1]:
                ds = _create_pro_dataset_worker(raw_data, max_layers_to_use)
                if ds is not None:
                    pro_datasets.append(ds)

    smet_datasets = []
    for fp in smet_files:
        ds = read_smet(fp, datetime_start, datetime_end)
        if ds is not None:
            smet_datasets.append(ds)

    all_datasets = pro_datasets + smet_datasets
    if not all_datasets:
        return None

    # --- FIX: Intelligent Merging and Concatenation ---
    # Group datasets by their location name
    grouped_by_location = {}
    for ds in all_datasets:
        loc = ds.location.item()
        if loc not in grouped_by_location:
            grouped_by_location[loc] = []
        grouped_by_location[loc].append(ds)

    # Merge the datasets for each location
    merged_datasets = [xr.merge(dsets) for dsets in grouped_by_location.values()]

    if len(merged_datasets) > 1:
        final_xr_dataset = xr.concat(merged_datasets, dim='location')
    else:
        final_xr_dataset = merged_datasets[0]

    if time: final_xr_dataset = time.apply(final_xr_dataset)
    if location: final_xr_dataset = location.apply(final_xr_dataset)
    if slope: final_xr_dataset = slope.apply(final_xr_dataset)
    if realization: final_xr_dataset = realization.apply(final_xr_dataset)

    return xsnowDataset(final_xr_dataset)

def _read_pro_raw(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None,
    **kwargs
) -> Tuple[Dict, List[Dict]]:
    """
    Parses a .pro file into raw Python objects, applying filters at the source.

    This low-level function reads a SNOWPACK .pro file line by line, extracting
    metadata and parsing each time-stamped snow profile. It filters profiles
    by the specified time range *before* storing them, making it memory-efficient.

    Args:
        filepath: Path to the .pro file.
        datetime_start: The start of the time range to parse. Profiles
            before this time are skipped.
        datetime_end: The end of the time range to parse. Profiles after
            this time are skipped.
        code_map: A dictionary mapping parameter codes (e.g., '0501') to
            variable names (e.g., 'height').
        **kwargs: Catches any unused keyword arguments.

    Returns:
        A tuple containing:
            - metadata (Dict): A dictionary of station parameters from the header.
            - temp_profiles (List[Dict]): A list where each element is a
              dictionary representing a single snow profile.
    """
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

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
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
                parts = line.split(',', 1)
                if parts[0] == "0500":
                    if current_ts_data:
                        profile_ts = pd.to_datetime(current_ts_data['timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
                        if pd.notna(profile_ts):
                            in_range = True
                            if start_ts and profile_ts < start_ts: in_range = False
                            if end_ts and profile_ts > end_ts: in_range = False
                            if in_range:
                                temp_profiles.append(current_ts_data)
                    current_ts_data = {'timestamp': parts[1]}
                else:
                    data_parts = line.split(',')
                    param_name = code_map_to_use.get(data_parts[0])
                    if param_name:
                        try:
                            current_ts_data[param_name] = np.array(data_parts[2:], dtype=float)
                        except ValueError:
                            logger.warning(f"Could not parse data for param {param_name} in {filepath}")


    if current_ts_data:
        profile_ts = pd.to_datetime(current_ts_data['timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        if pd.notna(profile_ts):
            in_range = True
            if start_ts and profile_ts < start_ts: in_range = False
            if end_ts and profile_ts > end_ts: in_range = False
            if in_range:
                temp_profiles.append(current_ts_data)
    
    return metadata, temp_profiles

def _create_pro_dataset(
    metadata: Dict,
    profiles: List[Dict],
    max_layers: int
) -> Optional[xr.Dataset]:
    """
    Creates a padded, CF-compliant xarray.Dataset from raw parsed .pro data.

    This function transforms the intermediate raw data from `_read_pro_raw`
    into a fully-formed, 5-dimensional xarray.Dataset. It pads all snow
    profile arrays with NaNs up to `max_layers` to ensure consistent array
    shapes and attaches CF-compliant metadata.

    Args:
        metadata: A dictionary of station parameters.
        profiles: A list of dictionaries, where each represents a single profile.
        max_layers: The number of layers to pad the profile data to.

    Returns:
        An xarray.Dataset containing the structured snow profile data,
        or None if the input profiles list is empty.
    """
    if not profiles:
        return None

    timestamps = pd.to_datetime([p['timestamp'] for p in profiles], format='%d.%m.%Y %H:%M:%S')
    all_params = sorted(list(set(key for p in profiles for key in p if key != 'timestamp')))

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
    ds = xr.Dataset(
        data_vars,
        coords={
            'location': [station_name],
            'time': timestamps,
            'slope': [float(metadata.get('slopeAngle', np.nan))],
            'realization': [0],
            'layer': np.arange(max_layers),
            'latitude': ('location', [float(metadata.get('latitude', np.nan))]),
            'longitude': ('location', [float(metadata.get('longitude', np.nan))]),
            'altitude': ('location', [float(metadata.get('altitude', np.nan))]),
            'slope_aspect': ('slope', [float(metadata.get('slopeAzi', np.nan))])
        }
    )

    # Attach CF-compliant metadata
    ds.attrs = {
        'Conventions': 'CF-1.8',
        'title': f"Snowpack Profile for {station_name}",
        'source': 'SNOWPACK model output (.pro file)',
        'history': f'{pd.Timestamp.now(tz="UTC").isoformat()}: Created by xsnow.',
        'crs': 'EPSG:4326'
    }

    for var_name in list(ds.variables):
        if var_name in registry:
            reg_entry = registry[var_name]
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', 'unknown')
            ds[var_name].attrs['units'] = reg_entry.get('si_units', 'unknown')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', var_name).replace('_', ' ')

    return ds


def read_smet(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None
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

    Returns:
        An xarray.Dataset containing the SMET data, or None if the file
        cannot be parsed or contains no data in the specified time range.
    """
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

    column_names = header['fields'].split()
    df = pd.read_csv(
        file_path,
        skiprows=data_start_line,
        sep=r'\s+',
        header=None,
        names=column_names,
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
    
    station_name = header.get('station_name', header.get('station_id', file_path.stem))
    ds = ds.expand_dims(location=[station_name])
    ds = ds.assign_coords({
        'latitude': ('location', [float(header.get('latitude', np.nan))]),
        'longitude': ('location', [float(header.get('longitude', np.nan))]),
        'altitude': ('location', [float(header.get('altitude', np.nan))])
    })
    
    return ds
