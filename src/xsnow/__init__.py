# src/xsnow/__init__.py
"""Top-level package for xsnow."""

from .core import xsnowDataset, DatasetDecorator
from .selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from .xsnow_io import read, append_latest, to_pro, to_smet, to_netcdf
from .plugin import register_xsnowDataset_method
from .registry import registry
from .sample_data import single_profile, single_profile_timeseries
from .combiners import combine, concat
from .utils import normalize_slope_coords, detect_slope_coord_issues, reorder_dims
from . import accessors
try:
    from ._version import version as __version__  # generated at build time by setuptools-scm
except Exception:
    __version__ = "0+unknown"


# Define what gets imported with 'from xsnow import *'
__all__ = [
    'xsnowDataset',
    'read',
    'append_latest',
    'to_pro',
    'to_smet',
    'to_netcdf',
    'single_profile',
    'single_profile_timeseries',
    'TimeSelector',
    'LocationSelector',
    'SlopeSelector',
    'RealizationSelector',
    'combine',
    'concat',
    'normalize_slope_coords',
    'detect_slope_coord_issues',
    'reorder_dims',

]
