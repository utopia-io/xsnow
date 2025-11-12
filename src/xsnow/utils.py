# utils.py
"""
This module provides common utility functions for the xsnow package.

These functions are designed to be used across different modules to reduce
code duplication and improve maintainability.
"""

import numpy as np
from typing import Union, Any, Iterable
import xarray as xr
import pandas as pd

from .core import xsnowDataset

def get_backend(array_like: Union[xsnowDataset, xr.Dataset, xr.DataArray]) -> Any:
    """
    Determines the appropriate numerical backend (NumPy or CuPy) for a given object.

    This function inspects the underlying data in an xarray object to see if it's
    a CuPy array (on a GPU) or a NumPy array (on the CPU) and returns the
    corresponding module.

    Args:
        array_like: An xsnowDataset, xarray.Dataset, or xarray.DataArray.

    Returns:
        The `numpy` or `cupy` module.
    """
    # If it's an xsnowDataset, get the underlying xarray.Dataset
    if isinstance(array_like, xsnowDataset):
        if array_like.data is None:
            return np
        data = array_like.data
    else:
        data = array_like

    # Find the first data variable to inspect
    if isinstance(data, xr.Dataset):
        first_var_name = next(iter(data.data_vars), None)
        if first_var_name is None:
            return np # No data variables, default to numpy
        array = data[first_var_name].data
    else: # It's a DataArray
        array = data.data

    # Check for the 'device' attribute, which is characteristic of a CuPy array
    if hasattr(array, 'device'):
        try:
            import cupy
            return cupy
        except ImportError:
            # This case is unlikely if a cupy array exists, but handled for safety
            return np
            
    return np


def concat(
    objs: Iterable[Union[xsnowDataset, xr.Dataset]],
    dim: Union[str, xr.DataArray, pd.Index],
    join: str = "outer",
    **kwargs,
):
    """
    xsnow-aware wrapper around xarray.concat.

    - Accepts an iterable of `xsnowDataset` and/or `xarray.Dataset`.
    - For join in {"outer", "exact"}:
        Behaves exactly like `xarray.concat`.
    - For join in {"inner", "left", "right"}:
        Applies the requested join semantics to all coordinates **except `layer`**.
        The `layer` dimension is always promoted to the union across inputs to
        avoid accidentally dropping layers.

    Parameters
    ----------
    objs : iterable of xsnowDataset or xarray.Dataset
        Datasets to concatenate.
    dim : str or DataArray or Index
        Dimension along which to concatenate (as in `xarray.concat`).
    join : {"outer", "inner", "left", "right", "exact"}, default "outer"
        Join behavior for non-concatenation coordinates.
        Note:
        - For "outer" and "exact": no xsnow-specific behavior is applied.
        - For "inner", "left", "right": 'layer' is treated specially and
          uses the union of layer coordinates from all inputs.
    **kwargs :
        Additional keyword arguments forwarded to `xarray.concat`.

    Returns
    -------
    xsnowDataset or xarray.Dataset
        Concatenated dataset. If any input was an `xsnowDataset`, the result is
        wrapped as `xsnowDataset`, otherwise a plain `xarray.Dataset` is returned.
    """
    objs = list(objs)
    if not objs:
        raise ValueError("xsnow.concat() received no datasets to concatenate.")

    # Unwrap xsnowDataset -> xarray.Dataset
    xr_objs = []
    saw_xsnow = False
    for obj in objs:
        if isinstance(obj, xsnowDataset):
            xr_objs.append(obj.data)
            saw_xsnow = True
        elif isinstance(obj, xr.Dataset):
            xr_objs.append(obj)
        else:
            raise TypeError(
                f"xsnow.concat only supports xsnowDataset or xarray.Dataset, got {type(obj)}"
            )

    if len(xr_objs) == 1:
        # trivial passthrough
        out = xr_objs[0]
        return xsnowDataset(out) if saw_xsnow else out

    # Fast path: for outer / exact, just delegate to xarray
    if join in ("outer", "exact"):
        out = xr.concat(xr_objs, dim=dim, join=join, **kwargs)
        return xsnowDataset(out) if saw_xsnow else out

    # For inner / left / right:
    # - apply requested join semantics to all dims
    # - but ensure `layer` uses the union of all its coordinates

    # Check if any dataset even has the layer dim; if not, just fall back.
    layer_dim = 'layer'
    has_layer = any(layer_dim in ds.dims for ds in xr_objs)
    if not has_layer:
        out = xr.concat(xr_objs, dim=dim, join=join, **kwargs)
        return xsnowDataset(out) if saw_xsnow else out

    # 1. Collect union of layer coords from ORIGINAL datasets
    all_layers = None
    for ds in xr_objs:
        if layer_dim in ds.dims:
            if layer_dim in ds.coords:
                vals = ds[layer_dim].values
            else:
                # if no coord, fall back to positional indices
                vals = np.arange(ds.sizes[layer_dim])
            all_layers = vals if all_layers is None else np.union1d(all_layers, vals)

    # Safety: shouldn't happen if has_layer is True, but guard anyway
    if all_layers is None:
        out = xr.concat(xr_objs, dim=dim, join=join, **kwargs)
        return xsnowDataset(out) if saw_xsnow else out

    # 2. Align all datasets once according to requested join
    #    This is lazy: it builds views / dask graphs, no compute().
    aligned = xr.align(*xr_objs, join=join)

    # 3. Reindex just the layer_dim to the union, for those that have it.
    #    This enforces the xsnow rule: never drop layers due to join semantics.
    reindexed = []
    for ds in aligned:
        if layer_dim in ds.dims:
            reindexed.append(ds.reindex({layer_dim: all_layers}))
        else:
            reindexed.append(ds)

    # 4. Final concat.
    #    Inputs are already aligned, so join here is mostly irrelevant.
    #    Use join="left" (or omit) to avoid a second alignment pass.
    out = xr.concat(reindexed, dim=dim, join="left", **kwargs)

    return xsnowDataset(out) if saw_xsnow else out
