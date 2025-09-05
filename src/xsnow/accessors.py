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
            as_dict (bool): If True, return the raw mapping dict (or {} if none exists).

        Returns:
            DataArray of string labels or dict with the integer-to-labels mapping
            (empty dict if no mapping exists).

        Raises:
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
        self._coding = self._obj.attrs.get("coding", None)

    def primary(self) -> xr.DataArray:
        """Primary ICSSG label"""
        if self._coding == 'slf_graincode':
            return slf_graincode_to_ICSSG_primary(self._obj)
        else:
            raise AttributeError("`coding` attribute missing or unknown. " +
                                 "You can only call this accessor on graincode variables that " +
                                 "specify a known code, e.g., 'slf_graincode'."
                                 )

    def secondary(self) -> xr.DataArray:
        """Secondary ICSSG label."""
        if self._coding == 'slf_graincode':
            return slf_graincode_to_ICSSG_secondary(self._obj)
        else:
            raise AttributeError("`coding` attribute missing or unknown. " +
                                 "You can only call this accessor on graincode variables that " +
                                 "specify a known code, e.g., 'slf_graincode'."
                                 )
