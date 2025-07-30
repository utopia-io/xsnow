import numpy as np
import xarray as xr
from xarray.core.extensions import register_dataarray_accessor

@register_dataarray_accessor("labels")
class LabelsAccessor:
    def __init__(self, xarray_obj):
        self._da = xarray_obj
        self._mapping = xarray_obj.attrs.get("mapping", None)
        if self._mapping is None:
            raise AttributeError("No 'mapping' attribute found in DataArray.")

    def get(self, as_dict=False) -> xr.DataArray | dict:
        """
        Return a labeled version of the DataArray using the integer-to-label mapping in 
        `.attrs["mapping"]`.

        Parameters:
            as_dict (bool): If True, returns a dict instead of a labeled DataArray.

        Returns:
            Labeled xarray.DataArray or dict.
        """
        if as_dict:
            return self._mapping
        vectorized = np.vectorize(self._mapping.get)
        return xr.apply_ufunc(
            vectorized,
            self._da,
            dask="allowed",
            output_dtypes=[str]
        )

# TODO: ZAccessor, TZAccessor
# @register_dataarray_accessor("z")
# @register_dataarray_accessor("tz")