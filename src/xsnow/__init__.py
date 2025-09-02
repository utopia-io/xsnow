# src/xsnow/__init__.py
"""Top-level package for xsnow."""

from .dataset import xsnowDataset
from .selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from .xsnow_io import read, update, to_pro, to_smet, to_netcdf
from .registry import registry
from .plugin import register_xsnowDataset_method
from . import accessors
from . import extensions


# Define what gets imported with 'from xsnow import *'
__all__ = [
    'xsnowDataset', 
    'register_xsnowDataset_method',
    'read',
    'update',
    'to_pro',
    'to_smet',
    'to_netcdf',
    'TimeSelector',
    'LocationSelector',
    'SlopeSelector',
    'RealizationSelector'
]