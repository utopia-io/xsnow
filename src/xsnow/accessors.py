import xarray as xr

@xr.register_dataarray_accessor("labels")
class LabelsAccessor:
    """
    A custom xarray accessor to retrieve the integer-to-string label mapping
    for a coordinate.
    
    This is used to get the human-readable names for coordinates that use
    integer indexing for efficiency.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get(self) -> dict:
        """
        Retrieves the label mapping dictionary from the coordinate's attributes.

        Returns:
            dict: The integer-to-name mapping, or an empty dictionary if no
                  mapping is found.
        """
        return self._obj.attrs.get("mapping", {})