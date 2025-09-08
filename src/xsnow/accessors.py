"""
Custom xarray accessors for human-readable labels.

This module provides two complementary accessors:

1. ``.labels`` accessor
   - Generic mechanism to attach and retrieve human-readable labels for any
     DataArray that uses integer codes internally.
   - Requires that the DataArray has an attribute ``.attrs["mapping"]``
     containing an integer-to-string dictionary.
   - Example:
       >>> import xarray as xr
       >>> da = xr.DataArray([0], dims="realization", name="realization")
       >>> da.attrs["mapping"] = {0: "manual profile"}
       >>> da.labels.get().item()
       'manual profile'

2. ``.icssg`` accessor
   - Specialized accessor for snow grain type codes according to the
     *International Classification for Seasonal Snow on the Ground* (ICSSG).
   - Operates on DataArrays that store three-digit SLF-style codes
     ("Swiss Code F1F2F3") in their ``.attrs["units"]`` metadata.
   - Provides convenience methods:
       * ``.icssg.primary()`` → returns primary ICSSG label
       * ``.icssg.secondary()`` → returns secondary ICSSG label

Together, these accessors allow xsnow datasets to remain efficient by storing
compact integer codes internally, while exposing user-friendly string labels
on demand for analysis, visualization, or export.
"""

import numpy as np
import xarray as xr
from xsnow.graincode_converter import slf_graincode_to_ICSSG_primary, slf_graincode_to_ICSSG_secondary

@xr.register_dataarray_accessor("labels")
class LabelsAccessor:
    """
    A custom xarray accessor to retrieve the human-readable names for variables
    that use integer codes for efficiency.

    This accessor will work on any coordinate or data variable that owns an attribute called
    `"mapping"`, which contains the integer-to-string dictionary.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._mapping = xarray_obj.attrs.get("mapping", None)

    def get(self, as_dict=False) -> xr.DataArray | dict:
        """
        Return a labeled version of the DataArray using the integer-to-label mapping stored
        in `.attrs["mapping"]`.

        Parameters:
        ----------
            as_dict (bool): If True, return the raw mapping dict (or {} if none exists).

        Returns:
        --------
            DataArray of string labels with dataset dimensions preserved or dict with the 
            integer-to-labels mapping (empty dict if no mapping exists).

        Examples:
        ---------
            >>> from xsnow.sample_data import single_profile
            >>> ds = single_profile()
            >>> ds["realization"]
            >>> ds["realization"].attrs["mapping"] = {0: "manual profile"}
            >>> ds["realization"].labels.get().item()
            'manual profile'

            >>> ds["location"].labels.get().item()
            'Kasererwinkl C6'

        Raises:
        -------
            AttributeError
                If no 'mapping' attribute is present **and** `as_dict=False`.
        """
        if self._mapping is None:
            if as_dict:
                return {}
            raise AttributeError("No 'mapping' attribute found on this DataArray.")

        if as_dict:
            return self._mapping

        def _lookup(v):
            try:
                if np.isnan(v):
                    return np.nan
            except Exception:
                pass
            if isinstance(v, np.integer):
                v = int(v)
            return self._mapping.get(v, None)

        return xr.apply_ufunc(
            np.vectorize(_lookup, otypes=[object]),
            self._obj,
            dask="allowed",
            output_dtypes=[object]
        )



@xr.register_dataarray_accessor("icssg")
class ICSSGAccessor:
    """
    An accessor to convert integer graincodes to human-readable grain type labels following the
    convention by the *International Classification for Seasonal Snow on the Ground* (ICSSG).

    Expected: DataArray of int/float codes; NaN allowed.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._unit = self._obj.attrs.get("units", None)

    def primary(self) -> xr.DataArray:
        """
        Convert an integer grain code to the ICSSG label of the **primary** grain type.

        Returns:
        --------
            DataArray of string labels broadcast over input dimensions.

        Examples:
        ---------
            >>> from xsnow.sample_data import single_profile
            >>> ds = single_profile()
            >>> ds["grain_type"].squeeze().values
            array([772.,   3.,   3., 772.,   4., 772.,   4.,   4.,   6.,   2.],
                dtype=float32)
            >>> ds["primary"] = ds["grain_type"].icssg.primary()
            >>> ds["primary"].squeeze().values
            array(['MFcr', 'RG', 'RG', 'MFcr', 'FC', 'MFcr', 'FC', 'FC', 'SH', 'DF'],
                dtype=object)

        Raises:
        -------
            AttributeError
                If the attribute in 'units' is not a known or implemented grain code
        """
        if self._unit == 'Swiss Code F1F2F3':
            return slf_graincode_to_ICSSG_primary(self._obj)
        else:
            raise AttributeError("Unknown grain code unit in `.attrs['units']`. "
                                 "Needs to be like `Swiss Code F1F2F3`."
                                 )

    def secondary(self) -> xr.DataArray:
        """
        Convert an integer grain code to the ICSSG label of the **secondary** grain type.

        Returns:
        --------
            DataArray of string labels broadcast over input dimensions.

        See Also:
        ---------
        primary : Convert to primary grain types (see Example section there).

        Raises:
        -------
            AttributeError
                If the attribute in 'units' is not a known or implemented grain code
        """
        if self._unit == 'Swiss Code F1F2F3':
            return slf_graincode_to_ICSSG_secondary(self._obj)
        else:
            raise AttributeError("Unknown grain code unit in `.attrs['units']`. "
                                 "Needs to be like `Swiss Code F1F2F3`."
                                 )
