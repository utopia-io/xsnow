import xarray as xr
from typing import Optional
from xsnow.registry import VARIABLE_REGISTRY

class xsnowDataset:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        # What do we need to do after each iteration of e.g., read_pro(),
        # and what has to be done only once, e.g., after read() is done with the last file?
        # -> we can sort that out later, for now:
        self._add_metadata()
        self._validate()

    def __getattr__(self, attr):
        # forward everything else to the underlying xr.Dataset (see postprocessing_extensions.md)
        return getattr(self.ds, attr)

    def _validate(self):
        """
        Ensure dimensions, tbd: required metadata/attributes, etc
        """
        required_dims = ['location', 'time', 'slope', 'realization', 'layer']
        for dim in required_dims:
            if dim not in self.ds.dims:
                raise ValueError(f"Dataset is missing required dimension: {dim}")

    def _add_metadata(self, registry=VARIABLE_REGISTRY):
        ds = self.ds
        for var, attrs in registry.items():
            if var in ds:
                ds[var].attrs.update(attrs)
