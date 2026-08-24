"""
Datetag assignment for snowpack layers

Datetags mark the notable storm dates of a season. Every layer is tagged
with the first datetag at or after its deposition date, so a tag labels
the snow that fell up to (and including) that storm. Datetag derivation
from weather timelines will live in this module as well.

References
----------
- Herla, F., et al. (2023). A large-scale validation of snowpack
  simulations in support of avalanche forecasting focusing on critical
  layers. Nat. Hazards Earth Syst. Sci., 23, appendix A.
"""
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import xsnowDataset
from ..graincode_converter import slf_graincode_to_ICSSG_primary
from ..plugin import register_xsnowDataset_method
from ..utils import get_logger, require_vars, wrap_like
from .hazard_chart import (
    _STABILITY_INDEX_INFO,
    _select_weak_and_deep,
    _take_layer,
    classify_poor_or_worse,
)
from .new_snow import NEW_SNOW_GRAINS

SHALLOW_OLD_SNOW = 'shallow_old_snow'
NEW_SNOW = 'new_snow'


def _as_day_tags(tags) -> np.ndarray:
    """Sorted day-resolution tag array."""
    if len(tags) == 0:
        return np.array([], dtype='datetime64[D]')
    return np.sort(pd.to_datetime(list(tags)).values.astype('datetime64[D]'))


def _tags_per_location(xs: xsnowDataset, datetags) -> list:
    """Resolve one sorted tag array per location."""
    if isinstance(datetags, (str, Path)):
        datetags = json.loads(Path(datetags).read_text())

    if not isinstance(datetags, Mapping):
        tags = _as_day_tags(datetags)
        return [tags] * xs.sizes['location']

    meta = datetags.get('meta')
    if not meta or 'level1' not in meta or 'level2' not in meta:
        raise ValueError(
            "Datetag mappings need a 'meta' entry naming the "
            "'level1'/'level2' location coordinates."
        )
    per_loc = []
    for level in (meta['level1'], meta['level2']):
        if level not in xs.data.coords:
            raise ValueError(
                f"Datetag mapping requires the location coordinate "
                f"'{level}', which is not present in the dataset."
            )
    l1 = np.atleast_1d(xs.data[meta['level1']].values)
    l2 = np.atleast_1d(xs.data[meta['level2']].values)
    missing = 0
    for a, b in zip(l1, l2):
        tags = datetags.get(str(a), {}).get(str(b))
        if tags is None:
            missing += 1
            tags = []
        per_loc.append(_as_day_tags(tags))
    if missing:
        get_logger(__name__).warning(
            f"No datetags for {missing} of {len(per_loc)} locations; "
            "their layers stay untagged."
        )
    return per_loc


@register_xsnowDataset_method
def assign_datetags(ds: 'xsnowDataset', datetags) -> 'xsnowDataset':
    """
    Tag every layer with the first datetag at or after its deposition date.

    Parameters
    ----------
    datetags
        Sequence of dates, a mapping in the AWSOME datetags.json shape
        (``{'meta': {'level1', 'level2'}, <l1>: {<l2>: [dates]}}``,
        resolved per location via the named location coordinates), or a
        path to such a JSON file.

    Returns
    -------
    xsnowDataset
        New dataset with the ``datetag`` variable (an existing one is
        overwritten). Layers deposited after the last tag, layers without
        a deposition date, and locations without tags are NaT.

    Notes
    -----
    Deposition dates are compared at day resolution.
    """
    logger = get_logger(__name__)
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(xs, ('deposition_date',), 'assign_datetags')

    dep_da = xs.data['deposition_date']
    loc_axis = dep_da.dims.index('location')
    days = dep_da.values.astype('datetime64[D]')

    out = np.full(days.shape, np.datetime64('NaT', 'ns'), 'datetime64[ns]')
    for i, tags in enumerate(_tags_per_location(xs, datetags)):
        if tags.size == 0:
            continue
        sel = [slice(None)] * days.ndim
        sel[loc_axis] = i
        sel = tuple(sel)
        d = days[sel]
        # NaT sorts last, so it lands out of range and stays NaT
        idx = np.searchsorted(tags, d.ravel(), side='left')
        tagged = np.where(
            idx < tags.size, tags[np.minimum(idx, tags.size - 1)],
            np.datetime64('NaT', 'D')
        )
        out[sel] = tagged.reshape(d.shape).astype('datetime64[ns]')

    if 'datetag' in xs.data:
        logger.info("Overwriting existing variable 'datetag'.")

    attrs = {'standard_name': 'deposition_datetag',
             'long_name': 'deposition datetag'}
    result = xs.data.assign(datetag=wrap_like(out, dep_da, attrs))
    return xs.__class__(result)


@register_xsnowDataset_method
def compute_simplified_stratigraphy(
    ds: 'xsnowDataset',
    indices: Sequence[str] | None = None,
    classes: dict[str, Sequence] | None = None,
) -> 'xsnowDataset':
    """
    Reduce the layer dimension to one entry per datetag.

    Groups are the assigned datetags (see ``assign_datetags``) plus
    ``shallow_old_snow`` (untagged, not new snow) and ``new_snow``
    (primary grain type PP or DF, any tag). Per group and stability
    index this keeps the weakest value, the depth of the deepest layer
    in the poor-or-worse classes (falling back to the weakest layer's
    depth), and the SLF grain codes of those two layers.

    Parameters
    ----------
    indices
        Layer variables to reduce; defaults to the variables present in
        ``_STABILITY_INDEX_INFO``.
    classes
        Per-index overrides for the ``_STABILITY_INDEX_INFO`` stability
        classes (ascending ``(label, upper_bound]`` steps, xsnow units).

    Returns
    -------
    xsnowDataset
        New dataset: non-layer variables are carried over; the new
        ``layer`` dimension holds one entry per group with the string
        coordinate ``datetag``. Per index: ``<idx>``, ``<idx>_depth``
        (m below surface), ``<idx>_gcode_weak``, ``<idx>_gcode_deep``,
        plus ``datetag_layer_count`` (0 = group absent, values NaN).
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(
        xs, ('datetag', 'height', 'HS', 'grain_type'),
        'compute_simplified_stratigraphy',
    )
    cfg = {
        k: (v['weak'], (classes or {}).get(k, v['classes']))
        for k, v in _STABILITY_INDEX_INFO.items()
    }
    if indices is None:
        indices = [v for v in cfg if v in xs.data]
    missing = [i for i in indices if i not in xs.data or i not in cfg]
    if missing:
        raise ValueError(
            f"No variable or class config for indices: {missing}"
        )

    height_da = xs.data['height']
    dims = tuple(d for d in height_da.dims if d != 'layer') + ('layer',)

    def _vals(da):
        return da.transpose(*dims).values

    height = _vals(height_da)
    depth = _vals(xs.data['HS'] - height_da)
    tags = _vals(xs.data['datetag'])
    gcode = _vals(xs.data['grain_type'])
    primary = _vals(slf_graincode_to_ICSSG_primary(xs.data['grain_type']))

    untagged = pd.isna(tags)
    tag_dates = np.unique(tags[~untagged])
    labels = [str(d)[:10] for d in tag_dates] + [SHALLOW_OLD_SNOW, NEW_SNOW]
    is_new = np.isin(primary, NEW_SNOW_GRAINS)
    valid = ~np.isnan(height)
    n_groups = len(tag_dates) + 2

    def group_mask(gi):
        """One group at a time; holding them all is the memory peak."""
        if gi < len(tag_dates):
            return tags == tag_dates[gi]
        if gi == len(tag_dates):
            return untagged & valid & ~is_new
        return is_new & valid

    base_shape = height.shape[:-1] + (n_groups,)
    count = np.zeros(base_shape, dtype='int16')
    out_vars: dict[str, np.ndarray] = {
        idx: np.full(base_shape, np.nan, dtype='float32')
        for i in indices
        for idx in (i, f'{i}_depth', f'{i}_gcode_weak', f'{i}_gcode_deep')
    }
    # a layer's stability class does not depend on its datetag group, so
    # both are resolved once per index instead of once per index and group
    prepared = {}
    for idx in indices:
        side, cls_steps = cfg[idx]
        values = _vals(xs.data[idx])
        prepared[idx] = (values, side,
                         classify_poor_or_worse(values, cls_steps))

    for gi in range(n_groups):
        mask = group_mask(gi)
        count[..., gi] = mask.sum(-1)
        for idx in indices:
            values, side, poor = prepared[idx]
            stab = np.where(mask, values, np.nan)
            w_idx, w_valid, pick = _select_weak_and_deep(
                stab, side, None, poor=poor & mask
            )
            out_vars[idx][..., gi] = np.where(
                w_valid, _take_layer(stab, w_idx), np.nan
            )
            out_vars[f'{idx}_depth'][..., gi] = np.where(
                w_valid, _take_layer(depth, pick), np.nan
            )
            out_vars[f'{idx}_gcode_weak'][..., gi] = np.where(
                w_valid, _take_layer(gcode, w_idx), np.nan
            )
            out_vars[f'{idx}_gcode_deep'][..., gi] = np.where(
                w_valid, _take_layer(gcode, pick), np.nan
            )

    out = xs.data.drop_dims('layer')
    new_dims = dims[:-1] + ('layer',)
    for idx in indices:
        attrs = dict(xs.data[idx].attrs)
        out[idx] = (new_dims, out_vars[idx], attrs)
        out[f'{idx}_depth'] = (
            new_dims, out_vars[f'{idx}_depth'],
            {'units': 'm', 'long_name': f'{idx} driving depth below surface'},
        )
        gattrs = {'units': 'Swiss Code F1F2F3'}
        out[f'{idx}_gcode_weak'] = (
            new_dims, out_vars[f'{idx}_gcode_weak'], dict(gattrs)
        )
        out[f'{idx}_gcode_deep'] = (
            new_dims, out_vars[f'{idx}_gcode_deep'], dict(gattrs)
        )
    out['datetag_layer_count'] = (new_dims, count)
    out = out.assign_coords(
        layer=np.arange(n_groups),
        datetag=('layer', labels),
    )
    return xs.__class__(out)
