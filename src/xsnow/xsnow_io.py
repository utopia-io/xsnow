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


def read(
    source: Union[str, List[str]],
    parallel: bool = True,
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    _code_map: Optional[Dict] = None
) -> Optional[xr.Dataset]:
    """
    High-level function to read one or more SNOWPACK files in parallel.

    This function acts as the main entry point for reading data. It can handle
    a single file path, a list of file paths, or a directory path. It auto-detects
    the file type (.pro, .smet, or .nc) and calls the appropriate parser. If both
    .pro and .smet files are provided, it ensures that SMET files only correspond
    to locations present in the .pro files, raising an error otherwise.

    Args:
        source (Union[str, List[str]]): A single file path, a directory path
            containing data files, or a list of file paths.
        parallel (bool): If True, reads files in parallel using all available
            CPU cores. Defaults to True.
        datetime_start (Optional[str]): If provided, the parser will only include
            data with timestamps on or after this date/time.
        datetime_end (Optional[str]): If provided, the parser will only include
            data with timestamps on or before this date/time.
        _code_map (Optional[Dict]): For internal/testing use.

    Returns:
        Optional[xr.Dataset]: A single, consolidated xarray.Dataset.
    """
    if isinstance(source, str):
        p = Path(source)
        if p.is_dir():
            filepaths = sorted(list(p.glob('*.pro'))) + sorted(list(p.glob('*.smet'))) + sorted(list(p.glob('*.nc')))
        elif p.is_file():
            if p.suffix == '.nc':
                return xr.open_dataset(p)
            filepaths = [p]
        else:
            logger.error(f"Source path does not exist: {source}")
            return None
    elif isinstance(source, list):
        filepaths = [Path(f) for f in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if not filepaths:
        logger.warning("No data files found in the specified source.")
        return None

    pro_files = [f for f in filepaths if f.suffix == '.pro']
    smet_files = [f for f in filepaths if f.suffix == '.smet']

    pro_datasets = []
    if pro_files:
        code_map_to_use = _code_map if _code_map is not None else CODE_TO_NAME_MAP
        if parallel and len(pro_files) > 1:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_ds = {
                    executor.submit(read_pro, fp, datetime_start, datetime_end, code_map_to_use): fp
                    for fp in pro_files
                }
                for future in concurrent.futures.as_completed(future_to_ds):
                    ds = future.result()
                    if ds is not None:
                        pro_datasets.append(ds)
        else:
            for fp in pro_files:
                ds = read_pro(fp, datetime_start, datetime_end, code_map_to_use)
                if ds is not None:
                    pro_datasets.append(ds)

    smet_datasets = []
    if smet_files:
        for fp in smet_files:
            ds = read_smet(fp, datetime_start, datetime_end)
            if ds is not None:
                smet_datasets.append(ds)

    if not pro_datasets and not smet_datasets:
        return None

    if pro_datasets and smet_datasets:
        pro_map = {ds.location.item(): ds for ds in pro_datasets}
        smet_map = {ds.location.item(): ds for ds in smet_datasets}
        
        pro_locations = set(pro_map.keys())
        smet_locations = set(smet_map.keys())
        
        mismatched_locs = smet_locations - pro_locations
        if mismatched_locs:
            raise ValueError(
                f"Data integrity error: SMET file locations not found in PRO files: {mismatched_locs}"
            )
            
        merged_datasets = []
        for loc in sorted(list(pro_locations)):
            to_merge = [pro_map[loc]]
            if loc in smet_map:
                to_merge.append(smet_map[loc])
            merged_datasets.append(xr.merge(to_merge))
        
        return xr.concat(merged_datasets, dim='location')
    
    elif pro_datasets:
        pro_datasets.sort(key=lambda ds: ds.coords['location'].item())
        return xr.concat(pro_datasets, dim='location')
    
    elif smet_datasets:
        smet_datasets.sort(key=lambda ds: ds.coords['location'].item())
        return xr.concat(smet_datasets, dim='location')

    return None


def read_smet(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None
) -> Optional[xr.Dataset]:
    """
    Parses a single SMET file and returns an xarray.Dataset.
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
    Parses a single SNOWPACK .pro file and returns a multi-dimensional xarray.Dataset.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return None

    code_map_to_use = code_map if code_map is not None else CODE_TO_NAME_MAP
    metadata: Dict[str, Any] = {}
    temp_profiles: List[Dict] = []
    current_ts_data: Dict = {}
    in_header, in_data = False, False

    start_ts = pd.to_datetime(datetime_start) if datetime_start else None
    end_ts = pd.to_datetime(datetime_end) if datetime_end else None

    with open(filepath, 'r') as f:
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

    if not temp_profiles:
        return None

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
