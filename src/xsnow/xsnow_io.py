# xsnow_io.py
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
import io

import numpy as np
import pandas as pd
import xarray as xr

# Assuming registry.py is in the same package or a reachable path
from .registry import registry

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

    Args:
        filepath (Path): Path to the .pro file.
        datetime_start (Optional[str]): Start timestamp for filtering profiles.
        datetime_end (Optional[str]): End timestamp for filtering profiles.
        code_map (Dict): Mapping from parameter codes to variable names.

    Returns:
        Tuple[Dict, List[Dict]]: A tuple containing the station metadata and a
            list of raw profile dictionaries.
    """
    metadata, temp_profiles = _read_pro_raw(filepath, datetime_start, datetime_end, code_map)
    return metadata, temp_profiles

def _create_pro_dataset_worker(
    raw_data: Tuple[Dict, List[Dict]],
    max_layers: int
) -> Optional[xr.Dataset]:
    """
    Internal worker to convert raw parsed .pro data into a padded xarray.Dataset.

    This function is designed to be run in a separate process for parallel
    dataset creation.

    Args:
        raw_data (Tuple[Dict, List[Dict]]): A tuple of metadata and raw
            profile data from `_parse_pro_worker`.
        max_layers (int): The number of layers to pad the dataset to.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset for a single station, or None
            if no profiles were found.
    """
    metadata, profiles = raw_data
    if not profiles:
        return None
    return _create_pro_dataset(metadata, profiles, max_layers)


def _read_worker(filepath: Path, datetime_start: Optional[str], datetime_end: Optional[str], code_map: Dict) -> Optional[xr.Dataset]:
    """
    Helper function to call the correct parser for a given file.

    Args:
        filepath (Path): Path to the data file.
        datetime_start (Optional[str]): Start timestamp for filtering.
        datetime_end (Optional[str]): End timestamp for filtering.
        code_map (Dict): Mapping from parameter codes to variable names.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the file's data.
    """
    if filepath.suffix == '.pro':
        # Note: This path is for single-file or non-parallel reads.
        # The main read function uses the two-stage workers for parallel pro reads.
        return read_pro(filepath, datetime_start, datetime_end, code_map)
    elif filepath.suffix == '.smet':
        return read_smet(filepath, datetime_start, datetime_end)
    elif filepath.suffix == '.nc':
        return xr.open_dataset(filepath)
    elif filepath.suffix == '.cro':
        return read_crocus(filepath, datetime_start, datetime_end)
    elif filepath.suffix == '.caaml':
        return read_caaml(filepath)
    elif filepath.suffix == '.json':
        return read_json(filepath)
    return None


def read(
    source: Union[str, List[str], Path],
    parallel: bool = True,
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    max_layers: Union[str, int] = 'auto',
    _code_map: Optional[Dict] = None
) -> Optional[xr.Dataset]:
    """
    High-level function to read one or more SNOWPACK-related files.

    This function serves as the main entry point for reading data into the xsnow
    package. It can handle various file formats (.pro, .smet, .nc, .zarr),
    read multiple files from a directory, and process them in parallel. It
    merges the data into a single, cohesive xarray.Dataset.

    Args:
        source (Union[str, List[str], Path]): The path to a file, a directory
            containing data files, or a list of file paths.
        parallel (bool, optional): If True, enables parallel processing for
            reading multiple .pro files. Defaults to True.
        datetime_start (Optional[str], optional): The start of the time range
            to read, in a pandas-parsable format (e.g., 'YYYY-MM-DD HH:MM:SS').
            Defaults to None, reading from the beginning.
        datetime_end (Optional[str], optional): The end of the time range to
            read. Defaults to None, reading to the end.
        max_layers (Union[str, int], optional): The maximum number of snow
            layers to allocate in the dataset for profile data. If 'auto',
            it is determined from the file(s) with the most layers.
            Defaults to 'auto'.
        _code_map (Optional[Dict], optional): Internal parameter to override
            the default mapping of .pro file parameter codes to names.
            Defaults to None.

    Returns:
        Optional[xr.Dataset]: A unified xarray.Dataset containing the data
            from all sources, or None if no valid data files are found.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            logger.error(f"Source path does not exist: {source}")
            return None
        
        # Optimized direct reads for single, non-processed file types
        if p.is_dir() and p.suffix == '.zarr':
            return read_zarr(p)
        if p.is_file() and p.suffix == '.nc':
            return read_netcdf(p)

        if p.is_dir():
            filepaths = (
                sorted(list(p.glob('*.pro'))) +
                sorted(list(p.glob('*.smet'))) +
                sorted(list(p.glob('*.nc'))) +
                sorted(list(p.glob('*.zarr')))
            )
        elif p.is_file():
            filepaths = [p]
        else: # Should be caught by p.exists() but is a safe fallback
            filepaths = []
            
    elif isinstance(source, list):
        filepaths = [Path(f) for f in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if not filepaths:
        logger.warning("No data files found in the specified source.")
        return None

    nc_files = [f for f in filepaths if f.suffix == '.nc']
    zarr_stores = [f for f in filepaths if f.suffix == '.zarr']
    pro_files = [f for f in filepaths if f.suffix == '.pro']
    smet_files = [f for f in filepaths if f.suffix == '.smet']

    datasets = []
    if nc_files:
        datasets.append(read_netcdf(nc_files))
    if zarr_stores:
        datasets.append(read_zarr(zarr_stores))

    code_map_to_use = _code_map if _code_map is not None else CODE_TO_NAME_MAP

    # --- Stage 1: Parallel Parsing of PRO files ---
    raw_pro_data = []
    if pro_files:
        if parallel and len(pro_files) > 1:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_data = {
                    executor.submit(_parse_pro_worker, fp, datetime_start, datetime_end, code_map_to_use): fp
                    for fp in pro_files
                }
                for future in concurrent.futures.as_completed(future_to_data):
                    raw_pro_data.append(future.result())
        else:
            for fp in pro_files:
                raw_pro_data.append(_parse_pro_worker(fp, datetime_start, datetime_end, code_map_to_use))

    # --- Stage 2: Determine Max Layers for PRO files ---
    if isinstance(max_layers, str) and max_layers == 'auto':
        if raw_pro_data:
            global_max = 0
            for _, profiles in raw_pro_data:
                if profiles:
                    local_max = max((len(p.get('height', [])) for p in profiles), default=0)
                    if local_max > global_max:
                        global_max = local_max
            max_layers_to_use = global_max if global_max > 0 else 1 # Avoid 0 layers
        else:
            max_layers_to_use = 200 # Default if no pro files
    elif isinstance(max_layers, int):
        max_layers_to_use = max_layers
    else:
        max_layers_to_use = 200 # Fallback default

    # --- Stage 3: Parallel Dataset Creation for PRO files ---
    pro_datasets = []
    if raw_pro_data:
        if parallel and len(raw_pro_data) > 1:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_ds = {
                    executor.submit(_create_pro_dataset_worker, raw_data, max_layers_to_use): raw_data
                    for raw_data in raw_pro_data
                }
                for future in concurrent.futures.as_completed(future_to_ds):
                    ds = future.result()
                    if ds is not None:
                        pro_datasets.append(ds)
        else:
            for raw_data in raw_pro_data:
                ds = _create_pro_dataset_worker(raw_data, max_layers_to_use)
                if ds is not None:
                    pro_datasets.append(ds)
    
    datasets.extend(pro_datasets)

    # --- Stage 4: Read SMET files (can also be parallelized) ---
    smet_datasets = []
    if smet_files:
        # This part can also be parallelized if SMET reading becomes a bottleneck
        for fp in smet_files:
            ds = read_smet(fp, datetime_start, datetime_end)
            if ds is not None:
                smet_datasets.append(ds)
    
    datasets.extend(smet_datasets)

    if not datasets:
        return None

    # Final merging and concatenation logic
    final_pro_datasets = [ds for ds in datasets if 'layer' in ds.dims]
    final_smet_datasets = [ds for ds in datasets if 'layer' not in ds.dims and 'time' in ds.dims]

    if final_pro_datasets and final_smet_datasets:
        pro_map = {ds.location.item(): ds for ds in final_pro_datasets}
        smet_map = {ds.location.item(): ds for ds in final_smet_datasets}
        
        pro_locations = set(pro_map.keys())
        smet_locations = set(smet_map.keys())
        
        mismatched_locs = smet_locations - pro_locations
        if mismatched_locs:
            raise ValueError(f"Data integrity error: SMET locations not in PRO locations: {mismatched_locs}")
            
        merged_datasets = []
        for loc in sorted(list(pro_locations)):
            to_merge = [pro_map[loc]]
            if loc in smet_map:
                to_merge.append(smet_map[loc])
            merged_datasets.append(xr.merge(to_merge))
        
        return xr.concat(merged_datasets, dim='location')
    
    all_datasets = final_pro_datasets + final_smet_datasets
    if len(all_datasets) > 1:
        try:
            all_datasets.sort(key=lambda ds: ds.coords['location'].item())
        except (AttributeError, KeyError):
            pass
        return xr.concat(all_datasets, dim='location')
    elif all_datasets:
        return all_datasets[0]

    return None

def _read_pro_raw(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None
) -> Tuple[Dict, List[Dict]]:
    """
    Parses a .pro file into raw Python objects without creating an xarray.Dataset.

    This is a low-level helper function that reads a SNOWPACK .pro file line by
    line, extracting metadata from the header and parsing each time-stamped
    snow profile from the data section. It returns data in a memory-efficient
    intermediate format, suitable for parallel processing.

    Args:
        filepath (Union[str, Path]): Path to the .pro file.
        datetime_start (Optional[str], optional): The start of the time range
            to parse. Profiles before this time are skipped. Defaults to None.
        datetime_end (Optional[str], optional): The end of the time range to
            parse. Profiles after this time are skipped. Defaults to None.
        code_map (Optional[Dict], optional): A dictionary mapping parameter
            codes (e.g., '0501') to variable names (e.g., 'height').
            If None, uses the default `CODE_TO_NAME_MAP`. Defaults to None.

    Returns:
        Tuple[Dict, List[Dict]]: A tuple containing:
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

    with open(file_path, 'r') as f:
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
                is_new_ts = parts[0] == "0500"

                if is_new_ts:
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
                        current_ts_data[param_name] = np.array(data_parts[2:], dtype=float)

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
    Creates a padded xarray.Dataset from raw parsed .pro data.

    This worker function takes the intermediate raw data from `_read_pro_raw`
    and transforms it into a fully-formed, 5-dimensional xarray.Dataset. It
    pads all snow profile arrays with NaNs up to a specified maximum number
    of layers to ensure consistent array shapes.

    Args:
        metadata (Dict): A dictionary of station parameters.
        profiles (List[Dict]): A list of dictionaries, where each dictionary
            represents a single snow profile.
        max_layers (int): The number of layers to pad the profile data to.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the structured
            snow profile data, or None if the input profiles list is empty.
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
                    # FIX: Ensure both source and destination are sliced correctly for truncation
                    layers_to_copy = min(num_layers, max_layers)
                    data_vars[param][1][0, i, 0, 0, :layers_to_copy] = np.array(values)[:layers_to_copy]

    station_name = metadata.get('stationName', 'unknown')
    lat = float(metadata.get('latitude', np.nan))
    lon = float(metadata.get('longitude', np.nan))
    alt = float(metadata.get('altitude', np.nan))
    slope_angle = float(metadata.get('slopeAngle', np.nan))
    slope_azi = float(metadata.get('slopeAzi', np.nan))

    ds = xr.Dataset(
        data_vars,
        coords={
            'location': [station_name],
            'time': timestamps,
            'slope': [slope_angle],
            'realization': [0],
            'layer': np.arange(max_layers),
            'latitude': ('location', [lat]),
            'longitude': ('location', [lon]),
            'altitude': ('location', [alt]),
            'slope_aspect': ('slope', [slope_azi])
        }
    )

    has_snow = ds['height'].notnull().any(dim='layer')
    ds['profile_status'] = has_snow.where(has_snow, "no_snow").where(~has_snow, "valid_profile_with_snow")
    ds.attrs['title'] = f"Snowpack Profile for {station_name}"
    ds['time'].attrs['long_name'] = 'time of profile'
    ds['layer'].attrs['long_name'] = 'layer index from bottom'
    ds['layer'].attrs['units'] = 'index'

    return ds


def read_netcdf(filepaths: Union[str, Path, List[Union[str, Path]]], 
                **kwargs
) -> Optional[xr.Dataset]:
    """
    Reads one or more NetCDF files into a single xarray.Dataset.

    This function is a wrapper around `xarray.open_dataset` and
    `xarray.open_mfdataset` for convenience. It can open a single NetCDF file
    or combine multiple files along their common dimensions.

    Args:
        filepaths (Union[str, Path, List[Union[str, Path]]]): The path to a
            single NetCDF file or a list of paths to multiple files.
        **kwargs: Additional keyword arguments passed directly to the
            underlying xarray open function.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the data from the
            NetCDF file(s), or None if an error occurs.
    """
    try:
        if isinstance(filepaths, (list, tuple)):
            return xr.open_mfdataset(filepaths, **kwargs)
        else:
            return xr.open_dataset(filepaths, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read NetCDF file(s): {e}")
        return None


def read_zarr(filepaths: Union[str, Path, List[Union[str, Path]]],
              **kwargs
) -> Optional[xr.Dataset]:
    """
    Reads one or more Zarr stores into a single xarray.Dataset.

    This function is a wrapper around `xarray.open_zarr` and
    `xarray.open_mfdataset` (with `engine='zarr'`). It can open a single Zarr
    store or combine multiple stores along their common dimensions.

    Args:
        filepaths (Union[str, Path, List[Union[str, Path]]]): The path to a
            single Zarr store (directory) or a list of paths to multiple stores.
        **kwargs: Additional keyword arguments passed directly to the
            underlying xarray open function.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the data from the
            Zarr store(s), or None if an error occurs.
    """
    try:
        if isinstance(filepaths, (list, tuple)):
            # For multiple zarr stores, open_mfdataset with engine='zarr' is needed
            return xr.open_mfdataset(filepaths, engine='zarr', **kwargs)
        else:
            return xr.open_zarr(filepaths, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read Zarr store(s): {e}")
        return None


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
        filepath (Union[str, Path]): Path to the SMET file.
        datetime_start (Optional[str], optional): The start of the time range
            to read. Data before this time is dropped. Defaults to None.
        datetime_end (Optional[str], optional): The end of the time range to
            read. Data after this time is dropped. Defaults to None.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the SMET data,
            or None if the file cannot be parsed or contains no data in the
            specified time range.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return None

    header = {}
    data_start_line = -1
    with open(file_path, 'r') as f:
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


def read_pro(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None
) -> Optional[xr.Dataset]:
    """
    Parses a single SNOWPACK .pro file into a multi-dimensional xarray.Dataset.

    This function provides a simple interface for reading a single .pro file. It
    acts as a wrapper, first calling the low-level raw parser `_read_pro_raw`
    and then using `_create_pro_dataset` to structure the data. The maximum
    number of layers is determined automatically from the profiles within the file.

    Args:
        filepath (Union[str, Path]): Path to the .pro file.
        datetime_start (Optional[str], optional): The start of the time range
            to read. Profiles before this time are skipped. Defaults to None.
        datetime_end (Optional[str], optional): The end of the time range to
            read. Profiles after this time are skipped. Defaults to None.
        code_map (Optional[Dict], optional): A dictionary to override the
            default mapping of parameter codes to names. Defaults to None.

    Returns:
        Optional[xr.Dataset]: An xarray.Dataset containing the snow profile
            data, or None if the file contains no valid profiles.
    """
    metadata, profiles = _read_pro_raw(filepath, datetime_start, datetime_end, code_map)
    if not profiles:
        return None
    max_layers = max((len(p.get('height', [])) for p in profiles), default=0)
    return _create_pro_dataset(metadata, profiles, max_layers)


def read_crocus(
    filepath: Union[str, Path], 
    datetime_start: Optional[str] = None, 
    datetime_end: Optional[str] = None
) -> Optional[xr.Dataset]:
    """
    Parses a single CROCUS file. Not yet implemented.
    """
    raise NotImplementedError("Reading CROCUS files is not yet implemented.")

def read_caaml(filepath: Union[str, Path]
) -> Optional[xr.Dataset]:
    """
    Parses a single CAAML file. Not yet implemented.
    """
    raise NotImplementedError("Reading CAAML files is not yet implemented.")

def read_json(filepath: Union[str, Path]
) -> Optional[xr.Dataset]:
    """
    Parses a single JSON file. Not yet implemented.
    """
    raise NotImplementedError("Reading JSON files is not yet implemented.")