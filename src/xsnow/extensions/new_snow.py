"""
New-snow amounts from stratigraphy and snow-height timelines
"""
import numpy as np

from ..core import xsnowDataset
from ..graincode_converter import (
    slf_graincode_to_ICSSG_primary,
    slf_graincode_to_ICSSG_secondary,
)
from ..plugin import register_xsnowDataset_method
from ..utils import require_vars
from .critical_crack_length import layer_thickness

NEW_SNOW_GRAINS = ('PP', 'DF')


@register_xsnowDataset_method
def compute_hn_ppdf(ds: xsnowDataset) -> xsnowDataset:
    """
    Total thickness of the layers whose primary or secondary grain type
    is PP or DF, the new-snow portion of the snow height.

    Returns
    -------
    xsnowDataset
        New dataset with the ``hn_ppdf`` variable [m]; needs the full-layer
        ``height`` and ``grain_type``.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(xs, ('height', 'grain_type'), 'compute_hn_ppdf')

    primary = slf_graincode_to_ICSSG_primary(xs.data['grain_type'])
    secondary = slf_graincode_to_ICSSG_secondary(xs.data['grain_type'])
    grains = list(NEW_SNOW_GRAINS)
    is_new = primary.isin(grains) | secondary.isin(grains)
    thickness = layer_thickness(xs.data['height'].fillna(np.nan))
    hn_ppdf = thickness.where(is_new, 0.0).sum('layer', min_count=0)

    attrs = {'units': 'm', 'long_name': 'thickness of PP/DF layers',
             'standard_name': 'new_snow_height_ppdf'}
    result = xs.data.assign(hn_ppdf=hn_ppdf.assign_attrs(attrs))
    return xs.__class__(result)


@register_xsnowDataset_method
def compute_hn(ds: xsnowDataset, window: int = 72,
               height_var: str = 'HS') -> xsnowDataset:
    """
    Snow-height increase over ``window`` hours [m]: ``height_var`` now
    minus ``window`` hours ago, clipped at 0.

    Parameters
    ----------
    window
        Whole number of hours; NaN where no sample exists exactly that far
        back.
    height_var
        Snow-height variable. ``HS``, the default, is the height xsnow
        derives from the layer heights on read and is computed here when
        missing; ``HS_mod`` is SNOWPACK's own timeseries.

    Returns
    -------
    xsnowDataset
        New dataset with the ``hn<window>`` variable (e.g. ``hn72``).
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    if height_var == 'HS' and 'HS' not in xs.data and 'height' in xs.data:
        xs = xs.compute_hs()
    require_vars(xs, (height_var,), 'compute_hn')
    if window != int(window) or window <= 0:
        raise ValueError(f"window must be a whole number of hours > 0, got {window!r}")
    window = int(window)

    hs = xs.data[height_var]
    lagged = hs.reindex(time=hs['time'] - np.timedelta64(window, 'h'))
    lagged = lagged.assign_coords(time=hs['time'])
    hn = (hs - lagged).clip(min=0.0)
    hn.attrs = {'units': 'm', 'long_name': f'{window} h snow height increase'}
    result = xs.data.assign({f'hn{window}': hn})
    return xs.__class__(result)
