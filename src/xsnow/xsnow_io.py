# xsnow_io.py
"""
Handles all data parsing and input/output operations for the xsnow package.

This module is the primary engine for reading and writing data in various
formats (e.g., SNOWPACK .pro, .smet, NetCDF).
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
    code_map: Dict
) -> Tuple[Dict, List[Dict]]:
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
    Reads, parses, and combines snowpack data into a unified xsnowDataset.

    This is the primary, user-facing function for loading data into the xsnow
    package. It is a high-level interface that discovers and reads snow profile
    (.pro) and time series (.smet) files from a given source, processes them
    (in parallel by default), and merges them into a single, coherent dataset.

    The function applies filters at the source for memory efficiency, allowing
    users to load only the specific time ranges or locations they need from
    large files.

    Args:
        source: The data source. Can be a path to a single file (e.g.,
            '.pro', '.smet', '.nc'), a directory containing data files, or a
            list of file paths.
        time: A `TimeSelector` object for filtering the data by time.
            When a slice is provided, filtering is performed at the source,
            which is highly memory-efficient.
        location: A `LocationSelector` object for filtering by station ID
            or geographic bounding box.
        slope: A `SlopeSelector` for filtering by slope angle.
        realization: A `RealizationSelector` for filtering by model realization.
        parallel: If True (default), multiple files will be read and processed
            concurrently using a process pool. Set to False for debugging.
        **parser_kwargs: Additional keyword arguments passed to the underlying
            parser functions. For example, `max_layers='auto'` or `max_layers=300`
            can be used to control the padding of snow profile data from .pro files.

    Returns:
        A unified `xsnowDataset` object containing the data from all sources,
        or `None` if no valid data files are found. The merging logic groups
        data by station, so if a .pro and .smet file are found for the same
        station, their data variables will be combined.

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
                raw_pro_data.extend([future.result() for future in concurrent.futures.as_completed(future_to_data)])
        else:
            raw_pro_data.extend(
                [_parse_pro_worker(fp, datetime_start, datetime_end, CODE_TO_NAME_MAP) for fp in pro_files]
            )

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

    grouped_by_location = {}
    for ds in all_datasets:
        loc = ds.location.item()
        if loc not in grouped_by_location:
            grouped_by_location[loc] = []
        grouped_by_location[loc].append(ds)

    merged_datasets = [
        xr.merge(dsets, compat='no_conflicts') for dsets in grouped_by_location.values()
        ]
    if not merged_datasets:
        return None

    # --- FIX: Build the location mapping and re-index datasets ---
    # 1. Get a sorted, unique list of all station names
    station_names = sorted([ds.location.item() for ds in merged_datasets])
    
    # 2. Create the mappings
    name_to_int = {name: i for i, name in enumerate(station_names)}
    int_to_name = dict(enumerate(station_names))

    reindexed_datasets = []
    for ds in merged_datasets:
        station_name = ds.location.item()
        station_id = name_to_int[station_name]
        # Assign the new integer coordinate to the 'location' dimension
        ds['location'] = [station_id]
        reindexed_datasets.append(ds)

    reindexed_datasets.sort(key=lambda ds: ds.location.item())
    
    if len(reindexed_datasets) > 1:
        # Use the reindexed datasets for concatenation
        final_xr_dataset = xr.concat(reindexed_datasets, dim='location')
    else:
        final_xr_dataset = reindexed_datasets[0]
        
    final_xr_dataset['location'].attrs['mapping'] = int_to_name

    if time: final_xr_dataset = time.apply(final_xr_dataset)
    if location: final_xr_dataset = location.apply(final_xr_dataset)
    if slope: final_xr_dataset = slope.apply(final_xr_dataset)
    if realization: final_xr_dataset = realization.apply(final_xr_dataset)

    # Create the final xsnowDataset object
    final_dataset = xsnowDataset(final_xr_dataset)

    # Calculate and add the 'z' coordinate if the dataset contains snow profile data before returning
    if 'height' in final_dataset.data:
        return final_dataset.compute_z_depth()
    else:
        return final_dataset

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
    metadata from the header and parsing each time-stamped snow profile from
    the data section. It returns data in a memory-efficient intermediate
    format, suitable for parallel processing. It filters profiles by the
    specified time range *before* storing them, which is crucial for
    memory efficiency when reading large files.

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
              dictionary representing a single snow profile at one timestamp.
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
                    if (param_name := code_map_to_use.get(data_parts[0])):
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
    try:
        ds['time'].tz.set_attr(int(header.get('tz')))
    except ValueError:
        logger.warning(f"Cannot set timezone from file {filepath}")

    station_name = header.get('station_name', header.get('station_id', file_path.stem))
    ds = ds.expand_dims(location=[station_name])
    ds = ds.assign_coords({
        'latitude': ('location', [float(header.get('latitude', np.nan))]),
        'longitude': ('location', [float(header.get('longitude', np.nan))]),
        'altitude': ('location', [float(header.get('altitude', np.nan))])
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
    if existing_ds.data is None or existing_ds.data.time.size == 0:
        new_data_obj = read(source)
        return new_data_obj or existing_ds

    last_timestamp = existing_ds.data.time.max().values
    last_timestamp_str = pd.to_datetime(last_timestamp).isoformat()
    
    time_filter = TimeSelector(sel=slice(last_timestamp_str, None))
    new_data_obj = read(source, time=time_filter)

    if new_data_obj is None or new_data_obj.data.time.size == 0:
        return existing_ds

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
def to_smet(ds: xsnowDataset, path: Union[str, Path], max_files: int = None, **kwargs):
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
        num_locations = ds.sizes['location']
        if num_locations > max_files:
            raise ValueError(
                f"Dataset contains {num_locations} locations, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )
            
    if ds.sizes['location'] > 1:
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
        f.write(f"nodata = -999\n")
        # SMET convention uses 'timestamp' as the first field
        fields = "timestamp " + " ".join(df.columns)
        f.write(f"fields = {fields}\n")
        f.write("[DATA]\n")

    # Append data, formatting timestamp correctly
    df.index.name = 'timestamp'
    df.to_csv(path, sep=' ', mode='a', header=False, date_format='%Y-%m-%dT%H:%M:%S')
    print(f"Dataset successfully saved to {path}")


@register_xsnowDataset_method
def to_pro(ds: xsnowDataset, path: Union[str, Path], max_files: int = None, **kwargs):
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
        num_profiles = ds.sizes['time']
        if num_profiles > max_files:
            raise ValueError(
                f"Dataset contains {num_profiles} profiles, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )

    if ds.sizes['location'] > 1:
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
            
            for var_name, code in NAME_TO_CODE_MAP.items():
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