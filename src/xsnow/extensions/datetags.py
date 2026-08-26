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
from typing import Optional
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
#: Thresholds of the datetag derivation, in the unit of the precipitation
#: variable: mm water equivalent per day, which is about cm of new snow.
DEFAULT_SNOW_THRESHOLDS = {
    'hn24_trace': 2.0, 'hn72_thresh': 10.0, 'hn72_increment': 5.0,
}
DEFAULT_RAIN_THRESHOLDS = {
    'rain24_trace': 5.0, 'rain72_thresh': 5.0, 'rain72_increment': 2.0,
}
DEFAULT_ADDITIONAL_TAGS = {'n_days': 10, 'hn24_cumsum': 25.0}
#: Air temperature [degC] separating snowfall from rainfall (SNOWPACK's
#: THRESH_RAIN) when the forcing carries no precipitation phase.
DEFAULT_TA_THRESHOLD = 1.2
_WINDOW_SLACK = 5   # days of slack for the storm rules
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


###############################################################################
# Datetag derivation from the weather timeline                                #
###############################################################################
# Ported from qmah (qmah/datetags/datetags.py, Avalanche Warning Service
# Salzburg / Florian Herla) so that the derivation runs on xsnow datasets
# instead of the NWP files and vstations tables of the legacy chain. The rules
# follow the appendix of Herla et al. (2023); the thresholds are unchanged.


def derive_datetags(
    wx: pd.DataFrame,
    human_datetags=None,
    snow_thresholds: Optional[Mapping] = None,
    rain_thresholds: Optional[Mapping] = None,
    additional_tags: Optional[Mapping] = None,
) -> np.ndarray:
    """
    Datetags of one spatial unit from its daily snow- and rainfall.

    A datetag marks a day on which a potentially critical layer was buried:
    the onset of a substantial storm following a dry spell, a day of
    substantial rain, or - where storms are far apart but small amounts keep
    accumulating - a day inserted inside the gap.

    Parameters
    ----------
    wx
        Daily weather of one spatial unit with the columns ``date``,
        ``hn24``, ``hn72``, ``rain24``, ``rain72``, sorted by date and
        indexed ``0..n-1`` (see :func:`daily_precipitation` and
        :func:`weather_datetags`). Amounts are in mm water equivalent.
    human_datetags
        Dates observed by forecasters, merged into the derived tags.
    snow_thresholds, rain_thresholds, additional_tags
        Overrides of :data:`DEFAULT_SNOW_THRESHOLDS`,
        :data:`DEFAULT_RAIN_THRESHOLDS`, :data:`DEFAULT_ADDITIONAL_TAGS`.

    Returns
    -------
    numpy.ndarray
        The datetags as ``datetime64[D]``.

    References
    ----------
    - Herla, F., Haegeli, P., Horton, S., & Mair, P. (2023). A large-scale
      validation of snowpack simulations in support of avalanche forecasting
      focusing on critical layers. NHESS, 23, appendix A.
    """
    snow = {**DEFAULT_SNOW_THRESHOLDS, **(snow_thresholds or {})}
    rain = {**DEFAULT_RAIN_THRESHOLDS, **(rain_thresholds or {})}
    extra = {**DEFAULT_ADDITIONAL_TAGS, **(additional_tags or {})}

    wx = wx.sort_values('date').reset_index(drop=True)
    if wx.empty:
        return np.array([], dtype='datetime64[D]')

    # A storm's onset is the last day below the trace amount before a 72 h
    # accumulation that both exceeds the threshold and rises clearly above
    # what that day already carried (otherwise the storm is receding).
    basetags_snow = _onset_tags(
        wx, 'hn24', 'hn72', snow['hn24_trace'], snow['hn72_thresh'],
        snow['hn72_increment'],
    )
    # Two tags a few days apart with little snow in between mark one storm.
    if len(basetags_snow) > 1:
        gaps = basetags_snow.diff().dt.total_seconds()[1:] / (24 * 3600)
        gaps = gaps.astype('int')
        between = np.array([
            wx.loc[
                (wx['date'] > basetags_snow.iloc[i])
                & (wx['date'] < basetags_snow.iloc[i + 1]), 'hn24'
            ].sum()
            for i in range(len(gaps))
        ])
        # of two tags belonging to one storm, drop the later one
        drop = np.where(
            (gaps < _WINDOW_SLACK) & (between < snow['hn72_thresh'])
        )[0] + 1
        basetags_snow = basetags_snow.drop(basetags_snow.index[drop])

    if human_datetags is not None:
        human = pd.to_datetime(list(human_datetags))
        human = human[pd.notna(human)]
        tags_snow = pd.to_datetime(sorted(set(basetags_snow) | set(human)))
    else:
        tags_snow = pd.to_datetime(sorted(set(basetags_snow)))

    tags_snow = _fill_snowfall_gaps(wx, tags_snow, extra)

    basetags_rain = _onset_tags(
        wx, 'rain24', 'rain72', rain['rain24_trace'], rain['rain72_thresh'],
        rain['rain72_increment'],
    )

    tags = sorted(set(tags_snow) | set(basetags_rain))
    return np.asarray(tags, dtype='datetime64[D]')


def _onset_tags(
    wx: pd.DataFrame,
    daily: str,
    accumulated: str,
    trace: float,
    threshold: float,
    increment: float,
) -> pd.Series:
    """Days below the trace amount that a substantial event follows."""
    quiet = wx[wx[daily] < trace].index
    onset = []
    for i, day in enumerate(quiet):
        # the event window reaches to the next quiet day, at most 5 days
        last = quiet[i + 1] - 1 if i < len(quiet) - 1 else len(wx)
        window = wx.loc[day + 1:min(day + 5, last), accumulated]
        if window.empty:
            continue
        peak = window.max()
        if peak > threshold and peak > wx.loc[day, accumulated] + increment:
            onset.append(day)
    return pd.to_datetime(wx.loc[onset, 'date'])


def _storm_end(
    wx: pd.DataFrame, tag: pd.Timestamp, next_tag: pd.Timestamp
) -> pd.Timestamp:
    """The day a storm has faded: the first day below a fifth of its peak."""
    horizon = min(next_tag, tag + pd.Timedelta(days=_WINDOW_SLACK))
    window = wx[(wx['date'] > tag) & (wx['date'] < horizon)]
    if window.empty:
        return tag
    peak_idx = window['hn24'].idxmax()
    after = wx[
        (wx['date'] > wx['date'].loc[peak_idx])
        & (wx['date'] < tag + pd.Timedelta(days=7))
    ]
    if after.empty:
        return wx['date'].loc[peak_idx]
    drier = after[after['hn24'] < 0.2 * wx.loc[peak_idx, 'hn24']]['date'].min()
    if not pd.isna(drier):
        return drier
    return after.loc[after['hn24'].idxmin(), 'date']


def _fill_snowfall_gaps(wx: pd.DataFrame, tags_snow, extra: Mapping):
    """Insert tags into long gaps whose small daily amounts still add up."""
    if len(tags_snow) < 2:
        return tags_snow
    spans = (tags_snow.diff().total_seconds()[1:] / (24 * 3600)).astype('int')
    ends = [
        _storm_end(wx, tags_snow[j], tags_snow[j + 1])
        for j in range(len(spans))
    ]
    accumulated = np.array([
        wx.loc[
            (wx['date'] > ends[j]) & (wx['date'] < tags_snow[j + 1]), 'hn24'
        ].sum()
        for j in range(len(spans))
    ])
    gaps = np.where(
        (spans > extra['n_days']) & (accumulated > extra['hn24_cumsum'])
    )[0]

    filler = []
    for gap in gaps:
        windows = int(min(
            np.round(accumulated[gap] / extra['hn24_cumsum']),
            np.round(spans[gap] / extra['n_days']),
        ))
        if windows <= 1:
            continue
        inside = (wx['date'] > ends[gap]) & (wx['date'] < tags_snow[gap + 1])
        hn24 = np.asarray(wx.loc[inside, 'hn24'])
        dates = np.asarray(wx.loc[inside, 'date'])
        if hn24.size < 3:
            continue
        rising = np.array([False] + list(np.diff(hn24) > 0) + [True])
        minima = np.where(np.diff(rising.astype('int')) > 0)[0]
        # Never tag the first or last day of the gap. Index 1 is excluded on
        # top of that to keep qmah's candidate set; index 0 is excluded where
        # qmah keeps it, because there qmah's own `forward[index - 1]` guard
        # below wraps to the whole-gap total and always passes.
        minima = minima[~np.isin(minima, np.array([1, len(hn24) - 1]))]
        minima = minima[(minima > 0) & (minima < len(hn24) - 1)]
        if minima.size == 0:
            continue
        # at least 10 mm of snow must remain on either side of the new tag
        forward = hn24.cumsum()
        backward = np.flip(np.flip(hn24).cumsum())
        minima = minima[
            (forward[minima - 1] > 10) & (backward[minima + 1] > 10)
        ]
        if minima.size == 0:
            continue
        for index in range(1, windows):
            # the tag that splits the gap most evenly in days and in snow
            cost_days = minima - (index * spans[gap] / windows)
            cost_snow = forward[minima] - (index * accumulated[gap] / windows)
            filler.append(dates[minima[np.argmin(abs(cost_days + cost_snow))]])
    if not filler:
        return tags_snow
    return pd.to_datetime(np.sort(np.concatenate(
        [np.asarray(tags_snow), pd.to_datetime(filler).values])))


def daily_precipitation(
    ds: 'xsnowDataset',
    ta_threshold: float = DEFAULT_TA_THRESHOLD,
    precip_var: str = 'PSUM',
    phase_var: str = 'PSUM_PH',
    snowfall_var: Optional[str] = None,
    rainfall_var: Optional[str] = None,
    scale: float = 1.0,
) -> xr.Dataset:
    """
    Daily snow- and rainfall of every location, plus their 72 h sums.

    Snowfall and rainfall come from an explicit pair of variables, from the
    precipitation phase when the forcing carries one, or from a temperature
    threshold on the air temperature.

    Parameters
    ----------
    ds
        Forcing dataset (``location``, ``time``); other dimensions are
        reduced to their first entry, the forcing being slope-independent.
    ta_threshold
        Air temperature [degC] below which precipitation falls as snow.
    precip_var, phase_var
        Precipitation and, if present, its liquid fraction.
    snowfall_var, rainfall_var
        Ready-made snowfall/rainfall variables, used in preference.
    scale
        Factor applied to the amounts, e.g. 3600 for a rate per second.

    Returns
    -------
    xarray.Dataset
        ``hn24``, ``rain24``, ``hn72``, ``rain72`` on (location, time), the
        time axis being daily.
    """
    data = ds.data if hasattr(ds, 'data') else ds
    for dim in ('slope', 'realization'):
        if dim in data.dims:
            data = data.isel({dim: 0}, drop=True)

    if snowfall_var and rainfall_var:
        snowfall, rainfall = data[snowfall_var], data[rainfall_var]
    else:
        if precip_var not in data:
            raise ValueError(
                f"Datetag derivation needs the precipitation variable "
                f"{precip_var!r}; the dataset has {sorted(data.data_vars)}."
            )
        precip = data[precip_var]
        if phase_var in data:
            phase = data[phase_var]
            if phase.isnull().any():
                if 'TA' not in data:
                    raise ValueError(
                        f"The precipitation phase {phase_var!r} has missing "
                        "values and the dataset carries no air temperature to "
                        "fill them from."
                    )
                phase = phase.fillna(
                    xr.where(data['TA'] >= ta_threshold, 1.0, 0.0))
            rainfall = precip * phase
            snowfall = precip - rainfall
        else:
            if 'TA' not in data:
                raise ValueError(
                    "Datetag derivation needs either a precipitation phase "
                    f"({phase_var!r}) or the air temperature to separate "
                    "snowfall from rainfall."
                )
            temperature = data['TA']
            units = str(temperature.attrs.get('units', '')).strip()
            if units == 'K' or (not units and float(temperature.min()) > 100.0):
                temperature = temperature - 273.15
            snowfall = precip.where(temperature < ta_threshold, 0.0)
            rainfall = precip.where(temperature >= ta_threshold, 0.0)

    hn24 = (snowfall * scale).resample(time='1D').sum()
    rain24 = (rainfall * scale).resample(time='1D').sum()
    if hn24.chunks is not None:      # resample leaves one chunk per day
        hn24 = hn24.chunk({'time': -1})
        rain24 = rain24.chunk({'time': -1})
    return xr.Dataset({
        'hn24': hn24,
        'rain24': rain24,
        'hn72': hn24.rolling(time=3, center=False).sum(),
        'rain72': rain24.rolling(time=3, center=False).sum(),
    })


@register_xsnowDataset_method
def weather_datetags(
    ds: 'xsnowDataset',
    groupby: Sequence[str] = ('region_id', 'band'),
    quantile: float = 0.5,
    human_datetags=None,
    ta_threshold: float = DEFAULT_TA_THRESHOLD,
    snow_thresholds: Optional[Mapping] = None,
    rain_thresholds: Optional[Mapping] = None,
    additional_tags: Optional[Mapping] = None,
    logger=None,
    **precipitation_kwargs,
) -> dict:
    """
    Derive the datetags of every spatial unit from a forcing dataset.

    Each unit - by default one elevation band of one region - is summarized
    by a quantile over its locations and tagged on its own, so that a storm
    that only hit the high alpine does not tag the valleys.

    Parameters
    ----------
    groupby
        The two location coordinates that span the spatial units. A missing
        coordinate collapses that level to ``'all'``.
    quantile
        Quantile across the locations of a unit (0.5 = the median location).
    human_datetags
        Dates observed by forecasters, merged into every unit's tags.
    **precipitation_kwargs
        Passed to :func:`daily_precipitation` (``precip_var``, ``phase_var``,
        ``snowfall_var``, ``rainfall_var``, ``scale``).

    Returns
    -------
    dict
        ``{'unique_datetags': [...], 'meta': {'level1', 'level2'},
        <level1>: {<level2>: [dates]}}`` - the shape
        :func:`assign_datetags` and the dashboard read.
    """
    logger = get_logger(__name__, logger=logger)
    if len(groupby) != 2:
        raise ValueError(
            "groupby names the two coordinates of a spatial unit, got "
            f"{list(groupby)}"
        )
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    daily = daily_precipitation(
        xs, ta_threshold=ta_threshold, **precipitation_kwargs
    )

    levels = []
    for name in groupby:
        if name in daily.coords and 'location' in daily[name].dims:
            levels.append(np.asarray(daily[name].values).astype(str))
        else:
            logger.info(
                "weather_datetags: no '%s' coordinate; the level collapses "
                "to 'all'", name,
            )
            levels.append(np.full(daily.sizes['location'], 'all'))

    daily = daily.compute()
    dates = pd.to_datetime(daily['time'].values)
    result: dict = {
        'meta': {'level1': str(groupby[0]), 'level2': str(groupby[1])},
    }
    unique: set = set()
    units = sorted(set(zip(levels[0], levels[1])))
    for level1, level2 in units:
        members = np.flatnonzero((levels[0] == level1) & (levels[1] == level2))
        unit = daily.isel(location=members)
        summary = unit.quantile(quantile, dim='location', skipna=True)
        frame = pd.DataFrame({
            'date': dates,
            'hn24': np.asarray(summary['hn24'].values, dtype='float64'),
            'rain24': np.asarray(summary['rain24'].values, dtype='float64'),
            'hn72': np.asarray(summary['hn72'].values, dtype='float64'),
            'rain72': np.asarray(summary['rain72'].values, dtype='float64'),
        })
        if frame[['hn24', 'rain24']].isna().all().any():
            logger.warning(
                "weather_datetags: %s/%s has no precipitation at all over %d "
                "location(s); it gets no datetags",
                level1, level2, members.size,
            )
        tags = derive_datetags(
            frame,
            human_datetags=human_datetags,
            snow_thresholds=snow_thresholds,
            rain_thresholds=rain_thresholds,
            additional_tags=additional_tags,
        )
        labels = [str(tag) for tag in tags]
        result.setdefault(level1, {})[level2] = labels
        unique.update(labels)
        logger.debug(
            "weather_datetags: %s/%s -> %d tag(s) from %d location(s)",
            level1, level2, len(labels), members.size,
        )
    logger.info(
        "weather_datetags: %d datetag(s) over %d spatial unit(s)",
        len(unique), len(units),
    )
    return {'unique_datetags': sorted(unique), **result}
