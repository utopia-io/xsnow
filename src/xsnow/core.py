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

from typing import Sequence, Union, Any, List, Optional
from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from .registry import registry
from .plugin import list_registered_methods
from .selectors import RealizationSelector

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
        if isinstance(source, xsnowDataset):
            self.data = source.data
        elif isinstance(source, xr.Dataset):
            self.data = source
        else:
            raise TypeError(
                "xsnowDataset must be initialized with an xarray.Dataset. "
                "To read from files, use the `xsnow.read()` function."
            )

    def __getstate__(self):
        """Support pickling for multiprocessing.
        
        Without this, __getattr__ causes infinite recursion during pickling
        when multiprocessing tries to serialize xsnowDataset objects.
        """
        return {'data': self.data}
    
    def __setstate__(self, state):
        """Support unpickling for multiprocessing."""
        self.data = state['data']
        
    def __repr__(self) -> str:
        """Provides a developer-friendly, summary representation of the dataset."""
        if self.data is None:
            return "<xsnowDataset (empty)>"
        
        try:
            locations = self.data.sizes.get('location', 0)
            times = self.data.sizes.get('time', 0)
            start_time = pd.to_datetime(self.data.time.min().values).strftime('%Y-%m-%d') if times > 0 else 'N/A'
            end_time = pd.to_datetime(self.data.time.max().values).strftime('%Y-%m-%d') if times > 0 else 'N/A'
            
            xr_repr = repr(self.data).splitlines()
            xr_repr[0] = f"employing the {xr_repr[0]}"
            indented_xr_repr = "\n".join("    " + line for line in xr_repr)

            return (
                f"<xsnowDataset>\n"
                f"  Locations:  {locations}\n"
                f"  Timestamps: {times} ({start_time}--{end_time})\n"
                f"  Profiles:   NotImplementedYet\n"  # @Ron, we could sum up the profile status
                                                      # tracker here to display sum of valid profiles
                                                      # with snow on the ground
                f"\n{indented_xr_repr}"
            )
        except Exception:
            return "<xsnowDataset with data (exception occurred on summary)>"

    def __getattr__(self, name: str):
        """
        Proxy to underlying xarray.Dataset. Extension methods are real
        attributes on xsnowDataset attached at import-time by the decorator.
        """
        if self.data is None:
            raise AttributeError(f"Dataset is not loaded. Cannot access '{name}'.")
        attr = getattr(self.data, name)
        if callable(attr):
            def method_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Re-wrap new xr.Dataset results so chaining stays in xsnowDataset world
                return self.__class__(result) if isinstance(result, xr.Dataset) else result
            return method_wrapper
        return attr

    def __dir__(self) -> Sequence[str]:
        """
        Return a list of attribute names available for tab completion.

        This extends the default attributes of xsnowDataset by also
        including the names of all variables in the underlying
        xarray.Dataset. Public attributes of the wrapped dataset
        are intentionally excluded to keep tab completion uncluttered.
        To explore xarrays methods users can fall back to the tab completion
        available at the self.data attribute.

        Returns
        -------
        list of str
            Sorted list of attribute names and data variable names.
        """
        base_attrs = set(super().__dir__())
        var_names = set()
        if self.data is not None:
            try:
                var_names = set(self.data.data_vars) | set(self.data.coords)
            except Exception:
                var_names = set()
        # we skip xr.Dataset’s public attributes to avoid clutter
        # xr_attrs = {a for a in dir(self.data) if not a.startswith("_")}

        # add registered extension method names
        base_attrs |= set(list_registered_methods().keys())

        # Combine the sets first, then convert every item to a string for sorting
        combined_attrs = base_attrs | var_names
        return sorted([str(item) for item in combined_attrs])


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

    def __contains__(self, key: str) -> bool:
        """
        Check whether a variable exists in the dataset.

        Parameters
        ----------
        key : str
            Name of the variable to test.

        Returns
        -------
        bool
            True if the variable exists in the underlying dataset,
            False otherwise.

        Raises
        ------
        KeyError
            If no dataset has been loaded into the ``xsnowDataset``.
        """
        if self.data is None:
            raise KeyError("Dataset is not loaded.")
        return key in self.data

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

# In core.py, inside the xsnowDataset class

    def merge_smet(
        self,
        source: Union[str, List[str], Path, List[Path]],
        var_prefix: Optional[str] = None,
        **kwargs
    ) -> 'xsnowDataset':
        """
        Reads SMET data and merges it into the dataset.

        An optional prefix can be added to the variable names from the SMET
        file to avoid naming conflicts (e.g., when merging input forcing
        data with model output data).

        Args:
            source: Path to a .smet file, a directory, or a list of files.
            var_prefix (Optional[str]): A string prefix to add to all data
                variable names read from the SMET files (e.g., 'input.').
            **kwargs: Additional keyword arguments passed to the underlying
                `xsnow.read()` function (e.g., time selectors).

        Returns:
            A new xsnowDataset with the combined data.
        """
        from xsnow.xsnow_io import read
        
        # If 'realization' is passed as an int, wrap it in a Selector object
        if 'realization' in kwargs and isinstance(kwargs['realization'], int):
            kwargs['realization'] = RealizationSelector(sel=kwargs['realization'])
            
        # Read the new SMET data, passing through any extra kwargs like time selectors
        smet_ds = read(source, **kwargs)

        # If there's no new data to merge, just return the original dataset
        if smet_ds is None or smet_ds.data is None:
            return self
        
        # If the original dataset is empty, just return the new data
        if self.data is None:
            return smet_ds

        smet_xarray = smet_ds.to_xarray()

        if var_prefix:
            # Create a mapping for renaming, e.g., {'TA': 'prefix.TA'}
            rename_mapping = {
                name: f"{var_prefix}{name}"
                for name in smet_xarray.data_vars
            }
            smet_xarray = smet_xarray.rename(rename_mapping)

        # Merge the (potentially renamed) SMET data into the existing dataset
        combined = xr.merge(
            [self.data, smet_xarray],
            compat='override',
            join='outer'
        )
        return xsnowDataset(combined)
            
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
        if self.data is None: 
            raise ValueError("Cannot compute HS on an empty dataset.")
        if 'height' not in self.data: 
            raise ValueError("Missing 'height' variable to compute HS.")
        
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
    
    def update_profile_status(
        self,
        data_vars_hint: list[str] | None = None,
        max_data_vars: int = 5,
    ) -> xr.Dataset:
        """
        Attach or update `profile_status` with the following semantics:

        < 0 : error / special flag (preserve as-is)
            0 : unknown / no information (yet)
            1 : valid profile WITHOUT snow
            2 : valid profile WITH snow

        Logic:
        - Start from existing profile_status if present, else from 0.
        - Set to 2 where snow is detected:
                * if `z` with `layer` dim exists: any(z < 0) along layer
                * elif no suitable `z` but `HS` exists: HS > 0
            (only when current status >= 0)
        - Set to 1 where:
                * status is still 0, AND
                * at least one selected non-status data variable with dims
                ⊇ (profile_status.dims) is non-null at that index.

        Candidate data variables for the 0→1 decision:
        - Up to `max_data_vars` are used.
        - Order of priority:
            1) `data_vars_hint` (filtered to existing vars with suitable dims),
            2) preferred vars: ['z', 'HS', 'height', 'density', 'temperature']
            (with suitable dims),
            3) all remaining data vars with suitable dims.
        """

        # -------------------------
        # 1. Base status
        # -------------------------
        if 'profile_status' in self:
            base = self['profile_status']
        else:
            core_dims = [d for d in ('location', 'time', 'slope', 'realization') if d in self.dims]
            if not core_dims:
                base = xr.DataArray(np.array(0, dtype='i1'), name='profile_status')
            else:
                coords = {d: self.coords[d] for d in core_dims}
                shape = [self.sizes[d] for d in core_dims]
                base = xr.DataArray(
                    np.zeros(shape, dtype='i1'),
                    coords=coords,
                    dims=tuple(core_dims),
                    name='profile_status',
                )

        # normalize dtype & NaNs → 0
        if base.dtype.kind == 'f':
            base = base.fillna(0).astype('i1')
        else:
            try:
                base = base.astype('i1')
            except TypeError:
                base = base.astype('float32').fillna(0).astype('i1')

        status = base

        # -------------------------
        # 2. Set 2 where snow is detected
        # -------------------------
        snow_mask = None

        if 'z' in self:
            z = self['z']
            if 'layer' in z.dims:
                snow_mask = (z < 0).any(dim='layer')

        if snow_mask is None and 'HS' in self:
            hs = self['HS']
            snow_mask = hs > 0

        if snow_mask is not None:
            snow_mask = snow_mask.broadcast_like(status)
            status = xr.where(snow_mask & (status >= 0), 2, status)

        # -------------------------
        # 3. Set 1 where status == 0 and we see data
        # -------------------------
        core_dims = status.dims
        candidates: list[str] = []

        def _suitable(var: xr.DataArray) -> bool:
            # Only use vars that are defined on at least all core dims
            return set(core_dims).issubset(var.dims)

        # 3.1 Hints first
        if data_vars_hint is not None:
            for name in data_vars_hint:
                if (
                    name in self.data_vars
                    and name != 'profile_status'
                    and name not in candidates
                    and _suitable(self.data_vars[name])
                ):
                    candidates.append(name)
                    if len(candidates) >= max_data_vars:
                        break

        # 3.2 Preferred list next
        preferred = ['z', 'HS', 'height', 'density', 'temperature']
        if len(candidates) < max_data_vars:
            for name in preferred:
                if (
                    name in self.data_vars
                    and name != 'profile_status'
                    and name not in candidates
                    and _suitable(self.data_vars[name])
                ):
                    candidates.append(name)
                    if len(candidates) >= max_data_vars:
                        break

        # 3.3 Fill remaining slots with any suitable data vars
        if len(candidates) < max_data_vars:
            for name, var in self.data_vars.items():
                if (
                    name != 'profile_status'
                    and name not in candidates
                    and _suitable(var)
                ):
                    candidates.append(name)
                    if len(candidates) >= max_data_vars:
                        break

        has_data = None

        for name in candidates:
            var = self[name]

            mask = var.notnull()
            extra_dims = [d for d in var.dims if d not in core_dims]
            if extra_dims:
                mask = mask.any(dim=extra_dims)

            mask = mask.broadcast_like(status)
            has_data = mask if has_data is None else (has_data | mask)

        if has_data is not None:
            status = xr.where((status == 0) & has_data, 1, status)

        # -------------------------
        # 4. Attributes & return
        # -------------------------
        attrs = dict(getattr(status, "attrs", {}))
        attrs.setdefault('long_name', 'profile status (error / no data / valid)')
        attrs.setdefault('units', '1')
        attrs.setdefault('standard_name', 'profile_status')
        attrs['mapping'] = {
            -1: "error during parsing",
            0: "unknown / no data",
            1: "valid, no snow",
            2: "valid, with snow",
        }
        status.attrs = attrs

        return self.__class__(self.data.assign(profile_status=status))


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
        if self.data is None: 
            return self
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
    



class DatasetDecorator(xsnowDataset):
    """
    A decorator that *is also* an xsnowDataset (subclassing), so it exposes the
    same API as the base wrapper, plus any extra methods/properties that will be added.
    """
    def __init__(self, base: "xsnowDataset", *, data: Optional[xr.Dataset] = None):
        self._base = base
        super().__init__(data if data is not None else base.data)
        
        # stash init args for consistent re-wrapping
        self._decorator_args   = getattr(base, "_decorator_args", (base,))
        self._decorator_kwargs = getattr(base, "_decorator_kwargs", {})
    
    @property
    def base(self):
        """Return the underlying xsnowDataset being decorated."""
        return self._base

    def _rewrap(self, new_data: xr.Dataset):
        """Return a new instance of *this* decorator class around new_data."""
        return self.__class__(*self._decorator_args, **self._decorator_kwargs, data=new_data)

    def __getattr__(self, name):
        # 1) Prefer the xsnowDataset surface (user methods, registered extensions, etc.)
        if hasattr(self.base, name):
            return getattr(self.base, name)

        # 2) Fall back to the underlying xr.Dataset API
        attr = getattr(self.data, name)
        if callable(attr):
            def method(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If an xarray.Dataset comes back, re-wrap it in the same decorator class
                return self._rewrap(result) if isinstance(result, xr.Dataset) else result
            return method
        return attr