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
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Union

import netCDF4
import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..units import validate_xsnow_unit_attrs
from ..utils import get_logger, _promote_slope_metadata_coords, reorder_dims

from .base import _resolve_archive_lazy, _round_coord
from .memory import MemoryLimitLike, check_memory_budget, resolve_memory_limit


@contextmanager
def _bounded_chunk_cache(cache_bytes: Optional[int]):
    """
    Clamp the netCDF4 per-variable chunk cache while opening or writing a file.

    Arguments:
        cache_bytes: Cache size in bytes, or None to leave netCDF4's
            current process-global setting untouched.
    """
    if cache_bytes is None:
        yield
        return
    previous = netCDF4.get_chunk_cache()
    _, nelems, preemption = previous
    netCDF4.set_chunk_cache(cache_bytes, nelems, preemption)
    try:
        yield
    finally:
        netCDF4.set_chunk_cache(*previous)


def _read_xsnow_netcdf(
    source: Union[str, Path],
    unit_validation: str = "strict",
    logger: Optional[logging.Logger] = None,
    lazy: Optional[bool] = None,
    chunks: Optional[Dict[str, int]] = None,
    time_sel=None,
    location_sel=None,
    slope_sel=None,
    realization_sel=None,
    chunk_cache_size: Optional[int] = None,
    memory_limit: MemoryLimitLike = "auto",
) -> Optional[xsnowDataset]:
    """
    Read xsnow's custom netcdf format.

    This includes proper json decoding of any 'mapping' attribute and
    handling of fixed-length byte coordinate strings.

    Args:
        source: Path to the NetCDF file.
        unit_validation: Stored-unit validation mode. `'strict'` raises when
            present unit attrs differ from xsnow units; `'light'` warns.
        logger: Optional logger for validation warnings.
        lazy: If True, return Dask-backed data backed by the open source file;
            the caller must close the returned dataset. If False, load and
            close the file before returning. If None (default), load selected
            data below the automatic archive-size threshold eagerly.
        chunks: Dask chunk sizes forwarded to ``xarray.open_dataset`` when
            ``lazy`` is not False. None uses xarray/Dask automatic chunking;
            ``{}`` requests the NetCDF backend's preferred chunks.
        time_sel: TimeSelector for filtering by time (already coerced).
        location_sel: LocationSelector for filtering by location (already coerced).
        slope_sel: SlopeSelector for filtering by slope (already coerced).
        realization_sel: RealizationSelector for filtering by realization
            (already coerced).
        chunk_cache_size: Per-variable netCDF4/HDF5 chunk cache in bytes
            while the file is open. None leaves the process's current netCDF4
            setting untouched; set a small value (e.g. 1 MiB) to limit
            aggregate chunk-cache memory on datasets with many variables.
        memory_limit: Pre-flight budget for an eager load. `'auto'` (default)
            uses a fraction of detected available memory; a size string or byte
            count sets an absolute cap; None disables the guard.

    Returns:
        An xsnowDataset instance, or None if reading fails.

    Raises:
        MemoryLimitExceeded: If an eager load would exceed ``memory_limit``.
    """
    logger = get_logger(__name__, logger=logger)
    budget_bytes = resolve_memory_limit(memory_limit, logger=logger)

    open_lazily = lazy is not False
    open_chunks = (
        (chunks if chunks is not None else "auto") if open_lazily else None
    )
    with _bounded_chunk_cache(chunk_cache_size):
        opened = xr.open_dataset(source, chunks=open_chunks)
    try:
        ds = opened

        # Decode fixed-length byte coords back to str (before the selectors run,
        # so string labels match)
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
        ds = _promote_slope_metadata_coords(ds)
        validate_xsnow_unit_attrs(
            ds,
            unit_validation=unit_validation,
            logger=logger,
            source=f"NetCDF file {source}",
        )

        # Apply selectors if provided (already coerced by read()), while the data
        # is still on disk, so an eager load only reads the requested subset
        if time_sel:
            ds = time_sel.apply(ds)
        if location_sel:
            ds = location_sel.apply(ds)
        if slope_sel:
            ds = slope_sel.apply(ds)
        if realization_sel:
            ds = realization_sel.apply(ds)

        resolved_lazy = _resolve_archive_lazy(ds, lazy)
        if lazy is None:
            logger.debug(
                f"Auto-detected archive lazy={resolved_lazy} from decoded size"
            )

        if not resolved_lazy:
            # Check the narrowed selection, not the whole file, and check it
            # before the allocation an OOM kill would make uncatchable.
            check_memory_budget(
                ds,
                budget_bytes,
                source=f"NetCDF file {source}",
                logger=logger,
            )
            ds.load()
            opened.close()

        return xsnowDataset(ds)
    except Exception:
        opened.close()
        raise


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
    compress: bool = True,
    complevel: int = 4,
    chunk_cache_size: Optional[int] = None,
    **kwargs
) -> None:
    """
    Saves the dataset to a NetCDF file.

    This method provides a convenient wrapper around the underlying
    `xarray.Dataset.to_netcdf()` method for easy caching and interoperability.

    The location mapping dictionary is converted to NetCDF-compatible format
    by encoding it as JSON in a string attribute.

    Args:
        ds: The dataset instance to save.
        path: The destination file path for the .nc file.
        logger: An optional preconfigured logger instance.
        compress: zlib-compress the data variables.
        complevel: zlib compression level (1-9) when compress is True.
        chunk_cache_size: Per-variable netCDF4/HDF5 chunk cache in bytes
            during the write. None leaves the process's current netCDF4
            setting untouched; set a small value (e.g. 1 MiB) to limit
            aggregate chunk-cache memory on datasets with many variables.
        **kwargs: Additional keyword arguments passed to `xarray.Dataset.to_netcdf()`.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    logger = get_logger(__name__, logger=logger)
    
    # Create a copy to avoid modifying the original
    ds_to_save = ds.to_xarray().copy()
        
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


    # Data variables are zlib-compressed by default. Snow datasets are dense
    # rectangular grids, but depending on the actual data they could be heavily
    # NA-padded. If a single model run produces some output the others don't
    # then this will be multiplied across all aspects, slopes, realizations, ...
    # which will all be nodata. A caller-supplied encoding takes precedence.
    if compress:
        caller_encoding = kwargs.pop("encoding", None) or {}
        encoding = {}
        for var_name in ds_to_save.data_vars:
            encoding[var_name] = {"zlib": True, "complevel": complevel}
        encoding.update(caller_encoding)  # caller overrides per variable
        kwargs["encoding"] = encoding

    path = Path(path)
    with _bounded_chunk_cache(chunk_cache_size):
        ds_to_save.to_netcdf(path, **kwargs)
    logger.info(f"Dataset successfully saved to {path}")
