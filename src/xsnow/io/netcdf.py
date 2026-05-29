# io/netcdf.py
"""
NetCDF I/O operations for xsnow.

This module handles reading and writing xsnow datasets in NetCDF format,
including proper JSON encoding/decoding for mapping attributes and
handling of string coordinates.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..utils import get_logger, reorder_dims

from .base import _round_coord


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
        del ds.attrs["mapping_json"]

    # variable-level
    for name, var in ds.variables.items():
        mj = var.attrs.get("mapping_json")
        if isinstance(mj, str):
            decoded = _parse(mj)
            var.attrs["mapping"] = decoded
            del var.attrs["mapping_json"]

    return reorder_dims(ds)


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
def to_netcdf(
    ds: xsnowDataset, 
    path: Union[str, Path],
    logger: Optional[logging.Logger] = None, 
    **kwargs
):
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
        ds: The dataset instance to save.
        path: The destination file path for the .nc file.
        logger: An optional preconfigured logger instance.
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

