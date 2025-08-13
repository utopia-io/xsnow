# xsnow_io.py
"""
This module contains all file parsers (e.g., read_pro) and writers for the
snowpack analysis package. It is responsible for translating raw data formats
into the standardized multi-dimensional xarray.Dataset used throughout the package.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import concurrent.futures

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


def read(source: Union[str, List[str]], parallel: bool = True, _code_map: Optional[Dict] = None) -> Optional[xr.Dataset]:
    """
    High-level function to read one or more SNOWPACK files in parallel.

    This function acts as the main entry point for reading data. It can handle
    a single file path, a list of file paths, or a directory path. It uses a
    ProcessPoolExecutor to read multiple files concurrently, which can
    significantly speed up data loading.

    Args:
        source (Union[str, List[str]]): A single file path, a directory path
            containing .pro files, or a list of file paths.
        parallel (bool): If True, reads files in parallel using all available
            CPU cores. Defaults to True.
        _code_map (Optional[Dict]): For internal/testing use. A map of parameter
            codes to names. If None, the default global map is used.

    Returns:
        Optional[xr.Dataset]: A single, consolidated xarray.Dataset containing
        the data from all files, merged along the 'location' dimension.
        Returns None if no valid files are found or an error occurs.
    """
    if isinstance(source, str):
        p = Path(source)
        if p.is_dir():
            filepaths = sorted(list(p.glob('*.pro')))
        elif p.is_file():
            filepaths = [p]
        else:
            logger.error(f"Source path does not exist: {source}")
            return None
    elif isinstance(source, list):
        filepaths = [Path(f) for f in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if not filepaths:
        logger.warning("No .pro files found in the specified source.")
        return None

    datasets = []
    # Use the provided code map, or fall back to the global default
    code_map_to_use = _code_map if _code_map is not None else CODE_TO_NAME_MAP

    if parallel and len(filepaths) > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Pass the code_map to each worker process
            future_to_ds = {executor.submit(read_pro, fp, code_map=code_map_to_use): fp for fp in filepaths}
            for future in concurrent.futures.as_completed(future_to_ds):
                ds = future.result()
                if ds is not None:
                    datasets.append(ds)
    else:
        # Read sequentially if parallel is False or only one file
        for fp in filepaths:
            ds = read_pro(fp, code_map=code_map_to_use)
            if ds is not None:
                datasets.append(ds)

    if not datasets:
        return None

    # Sort datasets by station name to ensure deterministic order before concatenating
    datasets.sort(key=lambda ds: ds.coords['location'].item())
    
    # Concatenate all individual datasets into a single one
    return xr.concat(datasets, dim='location')


def read_pro(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None
) -> Optional[xr.Dataset]:
    """
    Parses a single SNOWPACK .pro file and returns a multi-dimensional xarray.Dataset.

    Args:
        filepath (Union[str, Path]): Path to the .pro file.
        datetime_start (Optional[str]): Start date for filtering.
        datetime_end (Optional[str]): End date for filtering.
        code_map (Optional[Dict]): A map of parameter codes to names. If None,
            the default global map is used. This is the hook for testing.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return None

    # Use the provided code map, or fall back to the global default
    code_map_to_use = code_map if code_map is not None else CODE_TO_NAME_MAP

    # --- 1. Parsing Logic with Date/Time Optimization ---
    metadata: Dict[str, Any] = {}
    temp_profiles: List[Dict] = []
    current_ts_data: Dict = {}
    in_header, in_data = False, False

    start_ts = pd.to_datetime(datetime_start) if datetime_start else None
    end_ts = pd.to_datetime(datetime_end) if datetime_end else None

    with open(filepath, 'r') as f:
        for line in f:
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
                parts = line.split(',', 1)
                is_new_ts = parts[0] == "0500"

                if is_new_ts:
                    if current_ts_data:
                        profile_ts = pd.to_datetime(current_ts_data['timestamp'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
                        if pd.notna(profile_ts):
                            in_range = True
                            if start_ts and profile_ts < start_ts:
                                in_range = False
                            if end_ts and profile_ts > end_ts:
                                in_range = False
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
            if start_ts and profile_ts < start_ts:
                in_range = False
            if end_ts and profile_ts > end_ts:
                in_range = False
            if in_range:
                temp_profiles.append(current_ts_data)

    if not temp_profiles:
        logger.warning(f"No data was found in the specified range for file: {filepath}")
        return None

    # --- 2. Data Structuring (adapting to the 5D model) ---
    timestamps = pd.to_datetime([p['timestamp'] for p in temp_profiles], format='%d.%m.%Y %H:%M:%S')
    profiles = temp_profiles

    all_params = sorted(list(set(key for p in profiles for key in p if key != 'timestamp')))
    max_layers = max((len(p.get('height', [])) for p in profiles), default=0)

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
                    data_vars[param][1][0, i, 0, 0, :num_layers] = np.array(values)[:num_layers]

    # --- 3. Create the xarray.Dataset with 5D Coordinates ---
    station_name = metadata.get('stationName', Path(filepath).stem)
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

    ds.attrs['source_file'] = str(filepath)
    ds.attrs['title'] = f"Snowpack Profile for {station_name}"
    ds['time'].attrs['long_name'] = 'time of profile'
    ds['layer'].attrs['long_name'] = 'layer index from bottom'
    ds['layer'].attrs['units'] = 'index'

    return ds
