# utils.py
"""
This module provides common utility functions for the xsnow package.

These functions are designed to be used across different modules to reduce
code duplication and improve maintainability.
"""

import numpy as np
from typing import Union, Any
import xarray as xr

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
