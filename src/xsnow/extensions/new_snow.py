"""
New-snow amounts from stratigraphy and snow-height timelines
"""
import numpy as np

from ..core import xsnowDataset
from ..graincode_converter import slf_graincode_to_ICSSG_primary
from ..plugin import register_xsnowDataset_method
from ..utils import require_vars

NEW_SNOW_GRAINS = ('PP', 'DF')


@register_xsnowDataset_method
def compute_hn_ppdf(ds: xsnowDataset) -> xsnowDataset:
    """
    Total thickness of layers whose primary OR secondary grain type is
    PP or DF (the new-snow classification is wider here than for the
    ``new_snow`` datetag group, which uses the primary type only).

    Returns a new dataset with the ``hn_ppdf`` variable [m], the new-snow
    portion of the snow height. Runs on the full-layer dataset.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(xs, ('height', 'grain_type'), 'compute_hn_ppdf')

    from ..graincode_converter import slf_graincode_to_ICSSG_secondary
    from .critical_crack_length import layer_thickness

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
def compute_hn72(ds: xsnowDataset, height_var: str = 'HS_mod') -> xsnowDataset:
    """
    72 h snow-height increase [m]: ``height_var`` now minus 72 h ago,
    clipped at 0. NaN where no sample exists exactly 72 h earlier.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(xs, (height_var,), 'compute_hn72')

    hs = xs.data[height_var]
    lagged = hs.reindex(time=hs['time'] - np.timedelta64(72, 'h'))
    lagged = lagged.assign_coords(time=hs['time'])
    hn72 = (hs - lagged).clip(min=0.0)

    attrs = {'units': 'm', 'long_name': '72 h snow height increase'}
    result = xs.data.assign(hn72=hn72.assign_attrs(attrs))
    return xs.__class__(result)
