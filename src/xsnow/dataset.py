# dataset.py
"""
Defines the core xsnowDataset class for the xsnow package.

This module contains the primary user-facing object, `xsnowDataset`. It is
designed as a high-level, convenient wrapper around a standard `xarray.Dataset`
object, tailored for handling multi-dimensional snow profile data.

Core Design Philosophy
----------------------
The `xsnowDataset` class aims to simplify the user experience without
sacrificing the power and flexibility of the underlying xarray data model.
It achieves this through several key features:

1.  **Composition over Inheritance**: This class *contains* an `xarray.Dataset`
    instance (in `self.data`) rather than inheriting from it. This design
    choice is more robust and prevents future updates to `xarray` from
    breaking this class, giving us full control over the user-facing API.

2.  **Simplified Initialization**: The constructor is designed to accept a
    pre-existing `xarray.Dataset`. The primary, user-facing way to create a
    dataset from files is the top-level `xsnow.read()` function. This makes
    the extension architecture clean and powerful.

3.  **Seamless `xarray` Integration**: Through dunder forwarding (`__getattr__`,
    `__getitem__`, `__setitem__`), the class mimics the behavior of an
    `xarray.Dataset`, allowing users to interact with it using the familiar
    and powerful `xarray` API (e.g., `ds.sel(...)`, `ds['density']`).

4.  **Automatic Re-wrapping**: When a forwarded method returns a new
    `xarray.Dataset`, the class automatically wraps the result back into an
    `xsnowDataset` instance. This ensures that the user can chain operations
    (e.g., `ds.sel(...).mean(...)`) and always retain the convenience of the
    `xsnowDataset` wrapper.

Vertical Coordinate System
---------------------------
The package uses a physically intuitive vertical coordinate system.
- `z`: The primary vertical coordinate, representing depth below the snow
  surface in meters. The surface is z=0, and values are negative downwards.
- `height`: A data variable representing the original height above the
  ground in meters.
"""

from typing import Union, List, Any, TYPE_CHECKING
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

# Use TYPE_CHECKING to avoid circular imports at runtime for type hints
if TYPE_CHECKING:
    from xsnow.xsnow_io import read
    from xsnow.selectors import TimeSelector

from xsnow.registry import registry
# Import the registry from the plugin module
from xsnow.plugin import _analysis_plugins


class xsnowDataset:
    """
    A class to represent and interact with a multi-dimensional snow profile dataset.
    
    This class acts as a high-level wrapper around an xarray.Dataset, providing
    a convenient API for analysis, I/O, and data manipulation. It forwards
    unrecognized attribute and method calls to the underlying xarray.Dataset,
    enabling seamless use of the powerful xarray API.
    """

    def __init__(self, source: xr.Dataset):
        """
        Initializes the xsnowDataset from an existing xarray.Dataset.

        This is the low-level constructor for the class. The recommended,
        user-facing way to create an xsnowDataset from files is the top-level
        `xsnow.read()` function. This constructor is primarily used internally
        and by extension functions that return a new `xsnowDataset` instance.

        Args:
            source: An existing xarray.Dataset object.
        """
        if not isinstance(source, xr.Dataset):
            raise TypeError(
                "xsnowDataset must be initialized with an xarray.Dataset. "
                "To read from files, use the `xsnow.read()` function."
            )
        self.data: xr.Dataset = source

    def __repr__(self) -> str:
        """Provides a developer-friendly, summary representation of the dataset."""
        if self.data is None:
            return "<xsnowDataset (empty)>"
        
        try:
            locations = self.data.sizes.get('location', 0)
            times = self.data.sizes.get('time', 0)
            start_time = pd.to_datetime(self.data.time.min().values).strftime('%Y-%m-%d') if times > 0 else 'N/A'
            end_time = pd.to_datetime(self.data.time.max().values).strftime('%Y-%m-%d') if times > 0 else 'N/A'
            
            return (
                f"<xsnowDataset>\n"
                f"  Locations: {locations}\n"
                f"  Profiles: {times}\n"
                f"  Time Range: {start_time} to {end_time}"
            )
        except Exception:
            return f"<xsnowDataset with data>"

    def __getattr__(self, name: str) -> Any:
        """
        Forwards attribute access to a registered analysis plugin or the
        underlying xarray.Dataset.

        This is the core mechanism that enables the xsnowDataset to behave like
        an xarray.Dataset while also providing custom functionality. The lookup
        priority is:
        1. Registered analysis plugins (e.g., from `xsnow.register_xsnowDataset_method`).
        2. Attributes and methods of the underlying `xarray.Dataset` (`self.data`).

        If a forwarded method call returns a new `xarray.Dataset`, it is
        automatically re-wrapped in an `xsnowDataset` instance to allow for
        method chaining.
        """
        # Check the plugin registry first for the method name
        if name in _analysis_plugins:
            plugin_func = _analysis_plugins[name]
            def plugin_wrapper(*args, **kwargs):
                # Pass the instance (self) as the first argument
                result = plugin_func(self, *args, **kwargs)
                
                # FIX: Add re-wrapping logic for plugin results
                return self.__class__(result) if isinstance(result, xr.Dataset) else result
            return plugin_wrapper
        
        # If not in the registry, then try the underlying xarray.Dataset
        if self.data is None:
            raise AttributeError(f"Dataset is not loaded. Cannot access '{name}'.")
        
        attr = getattr(self.data, name)

        if callable(attr):
            def method_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return self.__class__(result) if isinstance(result, xr.Dataset) else result
            return method_wrapper
        else:
            return attr
                        
    def __getitem__(self, key: str) -> xr.DataArray:
        """
        Forwards item access to the underlying xarray.Dataset.

        This allows for dictionary-like access to data variables.

        Args:
            key: The name of the data variable to access.

        Returns:
            The corresponding xarray.DataArray.
        
        Examples
        --------
        >>> ds = xsnow.read('path/to/data')
        >>> # Access the 'density' DataArray
        >>> density_da = ds['density']
        """
        if self.data is None:
            raise KeyError("Dataset is not loaded.")
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        """
        Forwards item assignment to the underlying xarray.Dataset.

        This allows for dictionary-like assignment of new data variables.

        Args:
            key: The name of the new data variable.
            value: The xarray.DataArray or array-like object to assign.
        
        Examples
        --------
        >>> ds = xsnow.read('path/to/data')
        >>> # Create and assign a new variable
        >>> ds['new_var'] = xr.DataArray([1, 2, 3], dims=['z'])
        """
        if self.data is None:
            raise KeyError("Dataset is not loaded.")
        self.data[key] = value

    def copy(self, deep: bool = False) -> 'xsnowDataset':
        """
        Returns a copy of the dataset.

        This method provides a convenient wrapper around the underlying
        `xarray.Dataset.copy()` method and ensures the result is a new
        `xsnowDataset` instance.

        Args:
            deep: If True, the data and metadata are fully copied. If False
                (default), only the metadata is copied, and the new dataset's
                variables will still point to the original data buffers.

        Returns:
            A new xsnowDataset instance.
        
        Examples
        --------
        >>> ds = xsnow.read('path/to/data')
        >>> # Create a shallow copy (changes to ds_shallow affect ds)
        >>> ds_shallow = ds.copy(deep=False)
        >>> # Create a deep copy (a completely independent object)
        >>> ds_deep = ds.copy(deep=True)
        """
        if self.data is None:
            raise ValueError("Cannot copy an empty dataset.")
        copied_xr_dataset = self.data.copy(deep=deep)
        return self.__class__(copied_xr_dataset)

    def merge_smet(self, smet_source: Union[str, List[str], Path]) -> 'xsnowDataset':
        """
        Reads one or more SMET files and merges their data into this dataset.

        This method is useful for adding meteorological timeseries data to an
        existing dataset of snow profiles. The merge is performed by aligning
        the 'location' and 'time' coordinates.

        Args:
            smet_source: Path to a SMET file, a list of SMET files, or a
                directory containing SMET files.

        Returns:
            A new xsnowDataset instance with the merged data.
        
        Examples
        --------
        >>> ds = xsnow.read('station_A.pro')
        >>> ds_with_meteo = ds.merge_smet('station_A.smet')
        """
        from xsnow.xsnow_io import read

        if self.data is None:
            return self

        smet_data_obj = read(smet_source)
        if smet_data_obj is None:
            return self

        merged_ds = xr.merge([self.data, smet_data_obj.data], compat='no_conflicts')
        return self.__class__(merged_ds)
    
    def to_xarray(self) -> xr.Dataset:
        """
        Returns the underlying xarray.Dataset.

        This method unwraps the xsnowDataset and provides direct access to the
        core xarray.Dataset object. This is useful for interoperability with
        other libraries or functions that are built to work with standard
        xarray objects.

        Returns:
            The stored xarray.Dataset object.
        
        Examples
        --------
        >>> ds = xsnow.read('path/to/data')
        >>> # Get the raw xarray object for use with another library
        >>> xr_ds = ds.to_xarray()
        """
        return self.data

    def compute_hs(self) -> 'xsnowDataset':
        """
        Computes the total snow height (HS) from layer heights.

        This method calculates the maximum value of the 'height' variable along
        the 'layer' dimension for each profile. The result is added to the
        dataset as a new data variable named 'HS' (Total Snow Height).

        The resulting 'HS' variable will not have a 'layer' dimension.

        Returns:
            A new xsnowDataset instance with the 'HS' variable added.
            
        Raises:
            ValueError: If the dataset is empty or the 'height' variable is missing.

        Examples
        --------
        >>> ds = xsnow.read('path/to/data')
        >>> ds_with_hs = ds.compute_hs()
        >>> print(ds_with_hs['HS'])
        """
        if self.data is None: raise ValueError("Cannot compute HS on an empty dataset.")
        if 'height' not in self.data: raise ValueError("Missing 'height' variable to compute HS.")
        
        hs = self.data['height'].max(dim='layer', skipna=True)
        hs.attrs = {'long_name': 'Total Snow Height', 'units': 'm'}
        
        return self.__class__(self.data.assign({'HS': hs}))

    def compute_z_depth(self) -> 'xsnowDataset':
        """
        Calculates the depth below the surface (z) for each snowpack layer
        and adds it as a non-dimension coordinate.

        This method first computes the total snow height (HS) and then
        calculates z as (height - HS), adding it as a 'z' data
        coordinate to the dataset. Metadata is pulled from the central registry.

        Returns:
            A new xsnowDataset instance with the 'z' coordinate added.
        """
        if self.data is None:
            raise ValueError("Cannot compute on an empty dataset.")
        
        ds_with_hs = self.compute_hs() if 'HS' not in self.data else self
        
        z_depth = ds_with_hs.data['height'] - ds_with_hs.data['HS']
        
        # FIX: Pull metadata from the registry for consistency
        if 'z' in registry:
            reg_entry = registry['z']
            z_depth.attrs = {
                'long_name': reg_entry.get('long_name', 'Depth Below Snow Surface'),
                'units': reg_entry.get('si_units', 'm'),
                'standard_name': reg_entry.get('standard_name', 'depth'),
                'positive': 'up'
           }
        
        new_xr_ds = ds_with_hs.data.assign_coords(z=z_depth)
        
        return self.__class__(new_xr_ds)
    
    def to_gpu(self) -> 'xsnowDataset':
        """
        Moves the dataset's data to the GPU using CuPy if available.

        This method checks for an available GPU and the `cupy` library. If found,
        it converts the underlying NumPy arrays to CuPy arrays for accelerated
        computation. If `cupy` is not installed or a GPU is not available, it
        returns the original dataset without modification.

        Returns:
            A new xsnowDataset instance with data on the GPU, or the
            original object if a GPU is not available.
        
        Examples
        --------
        >>> ds_cpu = xsnow.read('path/to/data')
        >>> # Move to GPU for fast computation
        >>> ds_gpu = ds_cpu.to_gpu()
        """
        if self.data is None: return self
        try:
            import cupy as cp
            cp.arange(1) # Check if a GPU is present and cupy is working
            gpu_ds = self.data.copy(deep=False)
            for var in gpu_ds.data_vars:
                if np.issubdtype(gpu_ds[var].dtype, np.number):
                     gpu_ds[var].data = cp.asarray(gpu_ds[var].data)
            return self.__class__(gpu_ds)
        except (ImportError, RuntimeError):
            import logging
            logging.getLogger(__name__).info("Cupy not found or GPU not available. Returning original dataset.")
            return self
    def to_cpu(self) -> 'xsnowDataset':
        """
        Moves the dataset's data back to the CPU (NumPy).

        This is necessary before performing operations that are not supported
        on the GPU, such as saving to NetCDF. If the data is already on the
        CPU, it returns the original dataset.

        Returns:
            A new xsnowDataset instance with data on the CPU.
        
        Examples
        --------
        >>> ds_gpu = ds_cpu.to_gpu()
        >>> # ... perform GPU-accelerated analysis ...
        >>> ds_cpu_again = ds_gpu.to_cpu()
        >>> ds_cpu_again.to_netcdf('results.nc')
        """
        return self if self.data is None else self.__class__(self.data.as_numpy())