# xsnow/__init__.py
"""Top-level package for xsnow."""

from .dataset import xsnowDataset
from .selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from .xsnow_io import read, read_pro, read_netcdf, read_crocus, read_caaml, read_json, read_zarr
from .registry import registry
from .plugin import register_analysis

# Define what gets imported with 'from xsnow import *'
__all__ = ['xsnowDataset', 'register_analysis']