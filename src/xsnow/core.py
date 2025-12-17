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

    _REPR_MAX_AUTO_COMPUTE_SECONDS = 3
    _REPR_EST_BYTES_PER_SEC = 200 * 1024 * 1024  # heuristic throughput used for repr auto-compute
    _REPR_DASK_BLOCK_OVERHEAD = 0.005  # assumed per-block scheduler overhead for lazy arrays

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
        
    def __repr__(self, *, force_details: bool = False) -> str:
        """Provides a developer-friendly, summary representation of the dataset."""
        if self.data is None:
            return "<xsnowDataset (empty)>"
        
        try:
            locations = self.data.sizes.get('location', 0)
            times = self.data.sizes.get('time', 0)
            time_range = self._format_time_range(times, force_details=force_details)
            profile_summary, profile_note = self._format_profile_summary(force_details=force_details)
            
            xr_repr = repr(self.data).splitlines()
            xr_repr[0] = f"employing the {xr_repr[0]}"
            indented_xr_repr = "\n".join("    " + line for line in xr_repr)

            profiles_line = f"  Profiles:   {profile_summary}"
            if profile_note:
                indent = " " * len("  Profiles:   ")
                profiles_line = f"{profiles_line}\n{indent}{profile_note}"

            return (
                f"<xsnowDataset>\n"
                f"  Locations:  {locations}\n"
                f"  Timestamps: {times} ({time_range})\n"
                f"{profiles_line}\n"
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

    @staticmethod
    def _unwrap_xr(value: Union[xr.Dataset, "xsnowDataset"]) -> xr.Dataset:
        """
        Return the underlying ``xarray.Dataset`` from a dataset-like input.

        This exists mainly for internal convenience: extension code and helper
        methods sometimes already have an ``xsnowDataset`` (or decorated) object
        and sometimes a raw ``xarray.Dataset``. ``xsnow`` consistently stores
        the concrete data in ``.data`` as an ``xarray.Dataset``.
        """
        if isinstance(value, xr.Dataset):
            return value
        if isinstance(value, xsnowDataset):
            return value.data
        raise TypeError(
            "Expected an xarray.Dataset or xsnowDataset (decorated is fine); "
            f"got {type(value)!r}."
        )

    def _rewrap(self, new_data: Union[xr.Dataset, "xsnowDataset"]) -> "xsnowDataset":
        """
        Return a new wrapper instance around *new_data*.

        Notes
        -----
        This is a small internal helper used to keep wrapper types stable when
        an operation produces a new dataset. ``new_data`` can be either a raw
        ``xarray.Dataset`` or any ``xsnowDataset``-like wrapper; in the latter
        case its ``.data`` is used.
        """
        xr_ds = self._unwrap_xr(new_data)
        return self.__class__(xr_ds)

    def _format_time_range(self, times: int, force_details: bool) -> str:
        """Return a cheap time range summary if possible."""
        if times <= 0 or 'time' not in self.data:
            return "N/A"

        time_da = self.data['time']
        if not self._should_compute_repr_detail(time_da, force_details):
            return "lazy/uncomputed (use force_details=True)"

        try:
            start_da = self._maybe_compute(time_da.min())
            end_da = self._maybe_compute(time_da.max())
            start_time = self._to_datetime_str(start_da)
            end_time = self._to_datetime_str(end_da)
            return f"{start_time}--{end_time}"
        except Exception:
            return "unavailable"

    def _format_profile_summary(self, force_details: bool) -> tuple[str, Optional[str]]:
        """Return profile counts, avoiding heavy computation on lazy data."""
        try:
            profile_status = self.data['profile_status']
            total_profiles = int(profile_status.size)
        except Exception:
            return "unavailable", None

        lazy_detected = False
        est_seconds: Optional[float] = None

        if self._should_compute_repr_detail(profile_status, force_details):
            valid_profiles = self._compute_int_sum(profile_status >= 1)
            if valid_profiles is None:
                valid_profiles = "unavailable"
        else:
            lazy_detected = True
            est_seconds = self._estimate_repr_seconds(profile_status)
            valid_profiles = "<lazy/uncomputed>"

        try:
            hs = self.data['HS']
            if self._should_compute_repr_detail(hs, force_details):
                hs_mask = (hs > 0)
                try:
                    hs_mask = hs_mask.broadcast_like(profile_status)
                except Exception:
                    hs_mask, _ = xr.align(hs_mask, profile_status, join="outer")
                hs_positive_val = self._compute_int_sum(hs_mask)
                hs_positive = hs_positive_val if hs_positive_val is not None else "unavailable"
            else:
                lazy_detected = True
                est_hs = self._estimate_repr_seconds(hs)
                if est_seconds is None:
                    est_seconds = est_hs
                elif est_hs is not None:
                    est_seconds = max(est_seconds, est_hs)
                hs_positive = "<lazy/uncomputed>"
        except Exception:
            hs_positive = "unavailable"

        summary = f"{total_profiles} total | {valid_profiles} valid | {hs_positive} with HS>0"

        note = None
        if lazy_detected:
            est_str = f"{est_seconds:.2f} s" if est_seconds is not None else "? s"
            note = f"Compute profile sums with xsnowDataset.print(force_details=True) (estimated {est_str})"

        return summary, note

    def _should_compute_repr_detail(self, array: xr.DataArray, force_details: bool) -> bool:
        """Decide whether to compute a detail for repr based on laziness and size."""
        if force_details:
            return True
        if not self._is_lazy_dataarray(array):
            return True

        est_seconds = self._estimate_repr_seconds(array)
        if est_seconds is None:
            return False
        return est_seconds <= self._REPR_MAX_AUTO_COMPUTE_SECONDS

    def _maybe_compute(self, obj: Any) -> Any:
        """Compute lazy objects if possible, otherwise return as-is."""
        compute = getattr(obj, "compute", None)
        if callable(compute):
            try:
                return compute()
            except Exception:
                return obj
        return obj

    def _compute_int_sum(self, array: xr.DataArray) -> Optional[int]:
        """Compute a sum over a mask-like array, returning an int or None on failure."""
        try:
            summed = array.sum()
            summed = self._maybe_compute(summed)
            return self._to_int_scalar(summed)
        except Exception:
            return None

    def _to_int_scalar(self, value: Any) -> Optional[int]:
        """Extract an int from scalars, xarray, numpy, or dask objects."""
        try:
            return int(np.asarray(getattr(value, "values", value)).item())
        except Exception:
            try:
                return int(np.asarray(value).item())
            except Exception:
                return None

    def _to_datetime_str(self, value: Any) -> str:
        """Convert a datetime-like scalar to a YYYY-MM-DD string."""
        scalar = self._to_int_scalar(value)
        if scalar is None:
            try:
                arr = np.asarray(getattr(value, "values", value))
                scalar = arr.item()
            except Exception:
                scalar = None
        try:
            return pd.to_datetime(scalar).strftime('%Y-%m-%d')
        except Exception:
            return "unavailable"

    def _is_lazy_dataarray(self, array: xr.DataArray) -> bool:
        """Heuristic check for dask/lazy-backed arrays without importing dask."""
        data = getattr(array, "data", array)
        chunks = getattr(data, "chunks", None)
        if chunks not in (None, ()):
            return True

        module_name = getattr(type(data), "__module__", "").lower()
        if "dask" in module_name or "lazy" in module_name:
            return True

        return hasattr(data, "compute") and not isinstance(data, np.ndarray)

    def _estimate_repr_seconds(self, array: xr.DataArray) -> Optional[float]:
        """Estimate compute time from nbytes and chunk overhead for repr decisions."""
        data = getattr(array, "data", array)

        chunks = getattr(data, "chunks", None)
        nblocks = None
        if chunks not in (None, ()):
            try:
                nblocks = 1
                for axis_chunks in chunks:
                    nblocks *= len(axis_chunks)
            except Exception:
                nblocks = None

        try:
            dtype = getattr(array, "dtype", None)
            size = getattr(array, "size", None)
            nbytes = getattr(data, "nbytes", None)
            if nbytes is None and (dtype is not None and size is not None):
                nbytes = size * dtype.itemsize
            base = (nbytes / self._REPR_EST_BYTES_PER_SEC) if nbytes is not None else None
        except Exception:
            base = None

        if self._is_lazy_dataarray(array):
            overhead = (nblocks * self._REPR_DASK_BLOCK_OVERHEAD) if nblocks is not None else None
            if base is None and overhead is None:
                return None
            if base is None:
                return overhead
            if overhead is None:
                return base
            return base + overhead

        return base

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
        # Avoid conflicts on profile_status (timeseries SMETs don't carry layer info)
        if 'profile_status' in smet_xarray:
            smet_xarray['profile_status'] = smet_xarray['profile_status'] * np.nan

        try:
            smet_timezone = smet_xarray['time'].tz.get_attr()
        except KeyError:
            smet_timezone = None

        # Merge the (potentially renamed) SMET data into the existing dataset
        combined = xr.merge(
            [self.data, smet_xarray],
            compat='no_conflicts', combine_attrs="no_conflicts",
            join='outer'
        )
        # If base lacked timezone but SMET provided one, keep it
        if 'time' in combined and 'timezone' not in combined.time.attrs and smet_timezone is not None:
            combined.time.tz.set_attr(smet_timezone)

        return self.__class__(combined)
            
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
        
        height = self.data['height']
        # Handle empty layer dimension: if no layers, create HS full of NaN
        if height.sizes.get('layer', 0) == 0:
            core_dims = [d for d in height.dims if d != 'layer']
            coords = {d: height.coords[d] for d in core_dims}
            hs = xr.DataArray(np.full([height.sizes[d] for d in core_dims], np.nan, dtype=height.dtype),
                              coords=coords, dims=tuple(core_dims))
        else:
            hs = self.data['height'].max(dim='layer', skipna=True)

        # Attach metadata from registry when available, otherwise sensible defaults
        reg_entry = registry.get('HS', {})
        hs.attrs = {
            'standard_name': reg_entry.get('standard_name', 'height_of_snow'),
            'long_name': reg_entry.get('long_name', 'Total snow height'),
            'units': reg_entry.get('si_units', 'm'),
            'description': 'Total depth of snowpack (maximum height above ground)',
        }
        
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
    ) -> xr.Dataset:
        """
        Attach or update `profile_status` with very simple semantics:

            -1 : error during parsing  (preserved as-is)
            0 : no snow layer data
            1 : existing snow layer data (this may include "no snow", see below!)

        Logic
        -----
        1. Start from existing `profile_status` if present, otherwise from 0 over the
        core dims (location, time, slope, realization that exist in the dataset).

        **Negative values (< 0) are *never* changed by this method.**

        2. For `profile_status` to be positive after this call, the dataset must contain 
        the vertical (layer) variable `z` *and*
            * have negative `z` values (i.e., snow layers), *or*
            * NaN `z` values (i.e., *no* snow layers) but a pre-existing `profile_status==1`, *or*
            * NaN `z` values (i.e., *no* snow layers) but an existing `HS==0`.

        **Note that the profile_status counts *no snow* as existing layer information if the
        previous criteria are met.**

        If `z` doesn't exists, all non-negative profile_status entries are set to 0 (no layer data).
        Similarly, if `z` exists with all-NaN and `HS` does not exist or is NaN, the status
        is set to 0.
        """

        # -------------------------
        # 1. Base status
        # -------------------------
        if "profile_status" in self:
            base = self["profile_status"]
        else:
            core_dims = [d for d in ("location", "time", "slope", "realization") if d in self.dims]
            if not core_dims:
                base = xr.DataArray(np.array(0, dtype="i1"), name="profile_status")
            else:
                coords = {d: self.coords[d] for d in core_dims}
                shape = [self.sizes[d] for d in core_dims]
                base = xr.DataArray(
                    np.zeros(shape, dtype="i1"),
                    coords=coords,
                    dims=tuple(core_dims),
                    name="profile_status",
                )

        # normalize dtype & NaNs --> 0
        if base.dtype.kind == "f":
            base = base.fillna(0).astype("i1")
        else:
            try:
                base = base.astype("i1")
            except TypeError:
                base = base.astype("float32").fillna(0).astype("i1")

        status = base
        core_dims = status.dims  # e.g. ('location', 'time', 'slope', 'realization')
        mask_nonneg = status >= 0  # we never change negative codes

        # -------------------------
        # 3. No vertical variable --> everything non-negative becomes 0
        # -------------------------
        z_exists = True if 'z' in self and "layer" in self['z'].dims \
                        else False
        if not z_exists:
            status_out = xr.where(mask_nonneg, 0, status)
        else:
            v = self['z']
            # Reduce over all non-core, non-layer dims to get per-profile masks
            reduce_dims = [d for d in v.dims if d not in core_dims and d != "layer"]
            snowlike = v < 0
            if reduce_dims:
                snowlike = snowlike.any(dim=reduce_dims)
            snowlike = snowlike.any(dim="layer")
            snowlike = snowlike.broadcast_like(status)

            # HS-based fallback
            if "HS" in self and "layer" not in self["HS"].dims:
                hs = self["HS"]
                hs_reduce_dims = [d for d in hs.dims if d not in core_dims]
                hs_eq0 = (hs == 0)
                if hs_reduce_dims:
                    hs_eq0 = hs_eq0.any(dim=hs_reduce_dims)
                hs_eq0 = hs_eq0.broadcast_like(status)
            else:
                hs_eq0 = xr.zeros_like(status, dtype=bool)

            # Recompute status for all non-negative entries:
            # Start with 0 everywhere (for non-negative)
            status_out = xr.where(mask_nonneg, 0, status)

            # layer_var indicates something (z<0) --> 1
            status_out = xr.where(mask_nonneg & snowlike, 1, status_out)

            # No snowlike signal, but HS==0 at that profile or
            # pre-existing status was positive --> also 1
            status_out = xr.where(
                mask_nonneg & ~snowlike & (hs_eq0 | status>0),
                1,
                status_out,
            )

        # -------------------------
        # 4. Attributes & return
        # -------------------------
        attrs = dict(getattr(status_out, "attrs", {}))
        attrs.setdefault("long_name", "profile status (snow layer data)")
        attrs.setdefault("units", "1")
        attrs.setdefault("standard_name", "profile_status")
        attrs["mapping"] = {
            -1: "error during parsing",
            0: "no layer data",
            1: "existing layer data",
        }
        status_out.attrs = attrs

        # Reattach the original coordinate objects (with attrs) to avoid dropping
        # coord-level metadata during assignment.
        status_out = status_out.assign_coords(
            {d: self.data.coords[d] for d in status_out.dims if d in self.data.coords}
        )

        out = self.data.copy(deep=False)
        out["profile_status"] = status_out
        return self.__class__(out)
    

    def profile_status_summary(self) -> None:
        """Print a summary table of profile_status occurrences"""
        if self is not None and 'profile_status' in self:
            print(self['profile_status']
                .to_dataframe().groupby('profile_status')['profile_status'].count())
        else:
            print("No profile status available")



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
    Base class for extension "decorators" that layer functionality onto an ``xsnowDataset``.

    A ``DatasetDecorator`` is itself an ``xsnowDataset`` (it subclasses it) so it exposes
    the normal wrapper API (and the underlying ``xarray.Dataset`` via ``self.data``),
    but it also keeps a reference to another dataset wrapper in ``self.base``.

    The ``base`` reference is what makes the *decorator chain* work: a decorator can wrap
    either a plain ``xsnowDataset`` or another ``DatasetDecorator``. This allows multiple
    extensions to be stacked while still delegating higher-level functionality to the
    wrapped object.

    The keyword-only ``data`` parameter in ``__init__`` is intentionally *not* part of the
    public end-user API. It exists so internal machinery (primarily ``_rewrap`` and
    ``__getattr__`` forwarding) can create a new decorated instance around a new dataset
    produced by an xarray operation (e.g. ``sel``, ``drop_vars``, ``assign``) without
    mutating any existing instances.
    """
    def __init__(self, base: "xsnowDataset", *, data: Optional[Union[xr.Dataset, "xsnowDataset"]] = None):
        """
        Initialize a decorator instance.

        Parameters
        ----------
        base:
            The wrapped dataset object (either a plain ``xsnowDataset`` or another
            ``DatasetDecorator``). ``base`` is required so that:
              - attribute access can delegate to the already-configured wrapper chain
                (including previously applied extensions), and
              - ``_rewrap()`` can rebuild that chain around a new ``xarray.Dataset``
                when an xarray operation returns a new dataset.

        data:
            The concrete ``xarray.Dataset`` to expose as ``self.data`` for *this* wrapper
            instance. When omitted, the decorator uses ``base.data``.

            This is primarily an internal hook used by ``_rewrap()``: when an operation
            produces a new dataset, the decorator chain is rebuilt with ``base`` updated
            to wrap the new dataset, and ``data`` is passed explicitly so that the newly
            returned object consistently points at that same dataset.

            For convenience, ``data`` may also be an ``xsnowDataset`` (including another
            decorator instance). In that case its underlying ``.data`` is used.
        """
        self._base = base
        super().__init__(data if data is not None else base.data)
        
        # stash init args for consistent re-wrapping
        self._decorator_args   = getattr(base, "_decorator_args", (base,))
        self._decorator_kwargs = getattr(base, "_decorator_kwargs", {})
    
    @property
    def base(self) -> "xsnowDataset":
        """Return the underlying xsnowDataset being decorated."""
        return self._base

    def _rewrap(self, new_data: Union[xr.Dataset, "xsnowDataset"]):
        """
        Return a new instance of *this* decorator class around new_data.

        ``new_data`` can be either a raw ``xarray.Dataset`` or any ``xsnowDataset``-like
        wrapper; in the latter case its ``.data`` is used.

        This is primarily intended for extension authors. End users typically do not
        need to call ``_rewrap`` directly because xarray operations performed through
        the wrapper are automatically re-wrapped.

        Rebuild the base chain so ``base.data`` matches the coerced dataset (no mutation).
        """
        xr_ds = self._unwrap_xr(new_data)
        base = self._base

        if isinstance(base, DatasetDecorator):
            # Rebuild the entire decorator chain under us with new_data
            new_base = base._rewrap(xr_ds)
        elif isinstance(base, xsnowDataset):
            # Plain base wrapper → rebuild it around new_data
            new_base = base._rewrap(xr_ds)
        else:
            # Fallback (shouldn't happen in your design)
            new_base = xsnowDataset(xr_ds)

        # Create a fresh instance of *this* decorator with the rebuilt base
        new = self.__class__(new_base, data=xr_ds)

        # Optional: keep args/kwargs consistent for future rewraps
        new._decorator_args = (new_base,)
        new._decorator_kwargs = {}

        return new

    def __getattr__(self, name):
        # 1) Prefer the xsnowDataset surface (user methods, registered extensions, etc.)
        if hasattr(self.base, name):
            attr = getattr(self.base, name)
            if callable(attr):
                # If an inner decorator method returns an xsnowDataset-like wrapper,
                # promote it back to *this* outer decorator chain. If it returns a
                # plain xr.Dataset intentionally, keep it as-is.
                def method(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return self._rewrap(result) if isinstance(result, xsnowDataset) else result
                return method
            return attr

        # 2) Fall back to the underlying xr.Dataset API
        attr = getattr(self.data, name)
        if callable(attr):
            def method(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If an xarray.Dataset comes back, re-wrap it in the same decorator class
                return self._rewrap(result) if isinstance(result, xr.Dataset) else result
            return method
        return attr
