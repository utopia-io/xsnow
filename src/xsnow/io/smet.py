# io/smet.py
"""
SMET file I/O operations for xsnow.

SMET (Swiss Meteo Exchange Text) is a text-based format for meteorological
time series data. This module handles reading and writing SMET files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..registry import registry
from ..utils import get_logger

from .base import _open_any_as_text, _round_coord, _sanitize_station_name


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

    Slope metadata from the SMET header is stored as ``inclination`` and
    ``azimuth`` coordinates when available. If no slope metadata is present,
    the returned dataset contains one provisional slope with NaN slope
    coordinates. During a later ``xsnow.read()`` merge with profile data, such
    no-slope SMET data can be broadcast to all profile slopes at the same
    location.

    Args:
        filepath: Path to the SMET file.
        datetime_start: The start of the time range to read. Data before
            this time is dropped.
        datetime_end: The end of the time range to read. Data after
            this time is dropped.
        logger: An optional, pre-configured logger instance. If None, 
            the default module logger is used.

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
    with _open_any_as_text(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
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
    station_name = _sanitize_station_name(station_name)
    
    # Check if dimensions already exist before expanding
    dims_to_add = {}
    if 'location' not in ds.dims and 'location' not in ds.coords:
        dims_to_add['location'] = [station_name]
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
            
    # Add coordinates if they don't exist
    if 'latitude' not in ds.coords:
        ds = ds.assign_coords({
            'latitude': ('location', [_round_coord(header.get('latitude', np.nan))]),
            'longitude': ('location', [_round_coord(header.get('longitude', np.nan))]),
            'altitude': ('location', [float(header.get('altitude', np.nan))]),
        })
    
    # Try to get slope values from SMET header before defaulting to NaN
    if 'inclination' not in ds.coords:
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
    for var_name in ds.data_vars:
        if reg_entry := registry.get(str(var_name)):
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', str(var_name))
            ds[var_name].attrs['units'] = reg_entry.get('si_units', '1')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', str(var_name)).replace('_', ' ')
        else:
            # Set default attributes for variables not in registry
            if 'units' not in ds[var_name].attrs:
                ds[var_name].attrs['units'] = '1'
            if 'standard_name' not in ds[var_name].attrs:
                ds[var_name].attrs['standard_name'] = str(var_name)
            if 'long_name' not in ds[var_name].attrs:
                ds[var_name].attrs['long_name'] = str(var_name).replace('_', ' ')

    return ds


@register_xsnowDataset_method
def to_smet(
    ds: xsnowDataset, 
    path: Union[str, Path], 
    max_files: Optional[int] = None, 
    **kwargs
):
    """
    Saves time-series data from the dataset to a SMET file.

    This function extracts data variables that do not have a 'layer' dimension
    (e.g., meteorological data, total snow height) and writes them into the
    SMET format. It only supports writing data for a single location.

    Args:
        ds: The dataset instance to save.
        path: The destination file path for the .smet file.
        max_files: The maximum number of locations allowed.
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
    df = ds.data[smet_vars].squeeze(drop=True).to_dataframe()

    with _open_any_as_text(path, 'wt') as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        f.write(f"station_name = {ds.location.item()}\n")
        # Handle coordinates that may have extra dimensions (e.g., slope)
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
    
