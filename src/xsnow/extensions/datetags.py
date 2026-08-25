"""
Datetag assignment for snowpack layers

Datetags mark the notable storm dates of a season. Every layer is tagged
with the first datetag at or after its deposition date, so a tag labels
the snow that fell up to (and including) that storm. Datetag derivation
from weather timelines will live in this module as well.

Both routines run on blocks of complete profiles (see
:mod:`xsnow.extensions.profile_block_computation`), so Dask-backed
datasets stay lazy.

References
----------
- Herla, F., et al. (2023). A large-scale validation of snowpack
  simulations in support of avalanche forecasting focusing on critical
  layers. Nat. Hazards Earth Syst. Sci., 23, appendix A.
"""
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from ..core import xsnowDataset
from ..graincode_converter import slf_graincode_to_ICSSG_primary
from ..plugin import register_xsnowDataset_method
from ..utils import get_logger, require_vars
from .hazard_chart import (
    _STABILITY_INDEX_INFO,
    _select_weak_and_deep,
    _take_layer,
    classify_poor_or_worse,
)
from .new_snow import NEW_SNOW_GRAINS
from .profile_block_computation import apply_profile_block

SHALLOW_OLD_SNOW = 'shallow_old_snow'
NEW_SNOW = 'new_snow'
#: ``datetag`` attribute listing every tag assigned to the dataset
DATETAGS_ATTR = 'datetags'
_NAT_DAY = np.datetime64('NaT', 'D')
_NAT_INT = np.datetime64('NaT', 'D').astype('int64')


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


def _assign_datetags_block(deposition, tags):
    """Tag one block of profiles.

    ``deposition``: ``(..., layer)`` datetime64; ``tags``: ``(..., tag)``
    int64 days, sorted, NaT-padded, with size-1 axes where the tag rows do
    not vary (they only differ per location).
    """
    days = deposition.astype('datetime64[D]').astype('int64')
    tags = np.broadcast_to(tags, days.shape[:-1] + tags.shape[-1:])
    out = np.full(days.shape, _NAT_INT, dtype='int64')
    flat_days = days.reshape(-1, days.shape[-1])
    flat_tags = tags.reshape(-1, tags.shape[-1])
    flat_out = out.reshape(-1, days.shape[-1])
    rows, inverse = np.unique(flat_tags, axis=0, return_inverse=True)
    for i, row in enumerate(rows):
        valid = row[row != _NAT_INT]
        if valid.size == 0:
            continue
        sel = inverse.reshape(-1) == i
        d = flat_days[sel]
        idx = np.searchsorted(valid, d.ravel(), side='left')
        # an undated layer (NaT, the smallest int64) must stay NaT rather
        # than take the first tag; a layer after the last tag stays NaT too
        tagged = np.where(
            (idx < valid.size) & (d.ravel() != _NAT_INT),
            valid[np.minimum(idx, valid.size - 1)],
            _NAT_INT,
        )
        flat_out[sel] = tagged.reshape(d.shape)
    return out.astype('datetime64[D]').astype('datetime64[ns]')


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
        a deposition date, and locations without tags are NaT. The
        variable's ``datetags`` attribute lists every assigned tag, which
        :func:`compute_simplified_stratigraphy` uses as its group list.

    Notes
    -----
    Deposition dates are compared at day resolution. Dask-backed input
    stays lazy.
    """
    logger = get_logger(__name__)
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(xs, ('deposition_date',), 'assign_datetags')

    per_loc = _tags_per_location(xs, datetags)
    width = max((t.size for t in per_loc), default=0)
    table = np.full((len(per_loc), max(width, 1)), _NAT_INT, dtype='int64')
    for i, tags in enumerate(per_loc):
        table[i, :tags.size] = tags.astype('int64')
    tags_da = xr.DataArray(
        table, dims=('location', 'tag'),
        coords={'location': xs.data['location']},
    )
    all_tags = np.unique(np.concatenate(per_loc)) if per_loc else \
        np.array([], dtype='datetime64[D]')

    dep_da = xs.data['deposition_date']
    result = apply_profile_block(
        _assign_datetags_block, dep_da, tags_da,
        input_core_dims=[['layer'], ['tag']],
        output_dtype=np.dtype('datetime64[ns]'),
    )
    result = result.transpose(*dep_da.dims)
    result.attrs = {
        'standard_name': 'deposition_datetag',
        'long_name': 'deposition datetag',
        DATETAGS_ATTR: [str(d) for d in all_tags],
    }
    if 'datetag' in xs.data:
        logger.info("Overwriting existing variable 'datetag'.")
    return xs.__class__(xs.data.assign(datetag=result))


def _tag_days(datetag_da: xr.DataArray, logger) -> np.ndarray:
    """Sorted day tags present in the dataset, from the ``datetags``
    attribute or, failing that, a scan of the variable."""
    listed = datetag_da.attrs.get(DATETAGS_ATTR)
    if listed is not None:
        return _as_day_tags(list(listed))
    logger.info(
        "Variable 'datetag' carries no 'datetags' attribute; scanning it "
        "for the group list (assign_datetags records the list)."
    )
    values = datetag_da.data
    if isinstance(values, dask.array.Array):
        values = dask.array.unique(values).compute()
    values = np.asarray(values).ravel()
    return np.unique(values[~np.isnat(values)]).astype('datetime64[D]')


def _simplified_stratigraphy_block(
    height, hs, datetag, grain_code, is_new, *values, tag_ints, cfg,
):
    """Reduce one block of profiles to one entry per datetag group."""
    depth = hs[..., None] - height
    tags = datetag.astype('datetime64[D]').astype('int64')
    untagged = tags == _NAT_INT
    is_new = np.broadcast_to(is_new, height.shape)
    valid = ~np.isnan(height)
    n_groups = len(tag_ints) + 2

    def group_mask(gi):
        """One group at a time; holding them all is the memory peak."""
        if gi < len(tag_ints):
            return tags == tag_ints[gi]
        if gi == len(tag_ints):
            return untagged & valid & ~is_new
        return is_new & valid

    base_shape = height.shape[:-1] + (n_groups,)
    count = np.zeros(base_shape, dtype='int16')
    outs = [np.full(base_shape, np.nan, dtype='float32')
            for _ in range(4 * len(values))]
    # a layer's stability class does not depend on its datetag group, so
    # both are resolved once per index instead of once per index and group
    prepared = [
        (vals, side, classify_poor_or_worse(vals, cls_steps))
        for vals, (side, cls_steps) in zip(values, cfg)
    ]
    for gi in range(n_groups):
        mask = group_mask(gi)
        count[..., gi] = mask.sum(-1)
        for i, (vals, side, poor) in enumerate(prepared):
            stab = np.where(mask, vals, np.nan)
            w_idx, w_valid, pick = _select_weak_and_deep(
                stab, side, None, poor=poor & mask
            )
            outs[4 * i][..., gi] = np.where(
                w_valid, _take_layer(stab, w_idx), np.nan
            )
            outs[4 * i + 1][..., gi] = np.where(
                w_valid, _take_layer(depth, pick), np.nan
            )
            outs[4 * i + 2][..., gi] = np.where(
                w_valid, _take_layer(grain_code, w_idx), np.nan
            )
            outs[4 * i + 3][..., gi] = np.where(
                w_valid, _take_layer(grain_code, pick), np.nan
            )
    return (count, *outs)


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

    Notes
    -----
    The group list comes from the ``datetags`` attribute that
    ``assign_datetags`` records on ``datetag`` (so every assigned tag is a
    group, present or not); without it the variable is scanned for the
    tags present. Dask-backed input stays lazy.
    """
    logger = get_logger(__name__)
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

    tag_dates = _tag_days(xs.data['datetag'], logger)
    labels = [str(d)[:10] for d in tag_dates] + [SHALLOW_OLD_SNOW, NEW_SNOW]
    n_groups = len(labels)
    is_new = slf_graincode_to_ICSSG_primary(
        xs.data['grain_type']
    ).isin(list(NEW_SNOW_GRAINS))

    height_da = xs.data['height']
    inputs = [height_da, xs.data['HS'], xs.data['datetag'],
              xs.data['grain_type'], is_new]
    inputs += [xs.data[idx] for idx in indices]
    core = [['layer'], [], ['layer'], ['layer'], ['layer']]
    core += [['layer']] * len(indices)
    n_out = 1 + 4 * len(indices)
    results = apply_profile_block(
        _simplified_stratigraphy_block, *inputs,
        input_core_dims=core,
        output_core_dims=[['group']] * n_out,
        output_sizes={'group': n_groups},
        output_dtype=['int16'] + ['float32'] * (4 * len(indices)),
        kwargs={
            'tag_ints': tag_dates.astype('int64'),
            'cfg': [cfg[idx] for idx in indices],
        },
    )
    order = tuple(d for d in height_da.dims if d != 'layer') + ('layer',)

    def finish(da, attrs):
        da = da.rename({'group': 'layer'}).transpose(*order)
        da.attrs = attrs
        return da

    out = xs.data.drop_dims('layer')
    out['datetag_layer_count'] = finish(results[0], {})
    for i, idx in enumerate(indices):
        out[idx] = finish(results[4 * i + 1], dict(xs.data[idx].attrs))
        out[f'{idx}_depth'] = finish(
            results[4 * i + 2],
            {'units': 'm', 'long_name': f'{idx} driving depth below surface'},
        )
        gattrs = {'units': 'Swiss Code F1F2F3'}
        out[f'{idx}_gcode_weak'] = finish(results[4 * i + 3], dict(gattrs))
        out[f'{idx}_gcode_deep'] = finish(results[4 * i + 4], dict(gattrs))
    out = out.assign_coords(
        layer=np.arange(n_groups),
        datetag=('layer', labels),
    )
    return xs.__class__(out)
