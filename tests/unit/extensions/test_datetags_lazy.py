"""Lazy execution of assign_datetags and compute_simplified_stratigraphy."""
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

import xsnow
import xsnow.extensions.datetags  # noqa: F401 -- registers the methods
import xsnow.extensions.hazard_chart  # noqa: F401
from xsnow import xsnowDataset
from xsnow.extensions.datetags import DATETAGS_ATTR

DOMAIN_ARCHIVE = Path('/home/snowpack/zarrtest/domain8/output/domain8.zarr')
STATION_ARCHIVE = Path('/home/snowpack/zarrtest/output/x1.zarr')


@pytest.fixture
def season():
    """Two stations, two slopes, 40 timesteps, 30 layers, growing snowpack
    with deposition dates spread over the season."""
    rng = np.random.default_rng(3)
    L, S, T, K = 2, 2, 40, 30
    times = pd.date_range('2025-11-01', periods=T, freq='3D')
    height = np.full((L, T, S, 1, K), np.nan, np.float32)
    dep = np.full((L, T, S, 1, K), np.datetime64('NaT', 'ns'),
                  'datetime64[ns]')
    gtype = np.full((L, T, S, 1, K), np.nan, np.float32)
    sk38 = np.full((L, T, S, 1, K), np.nan, np.float32)
    for lo in range(L):
        for sl in range(S):
            for t in range(T):
                n = min(K, 1 + t * 3 // 4 + lo + sl)
                height[lo, t, sl, 0, :n] = np.cumsum(
                    rng.uniform(0.02, 0.08, n)
                ).astype(np.float32)
                dep[lo, t, sl, 0, :n] = (
                    times[0].to_datetime64()
                    + np.sort(rng.integers(0, 3 * t + 1, n)).astype(
                        'timedelta64[D]'
                    )
                )
                codes = rng.choice([110, 220, 330, 440, 550, 772], n)
                codes[-1] = 110      # fresh snow on top
                gtype[lo, t, sl, 0, :n] = codes
                sk38[lo, t, sl, 0, :n] = rng.uniform(0.3, 4.0, n)
    dims = ('location', 'time', 'slope', 'realization', 'layer')
    data = xr.Dataset(
        {
            'height': (dims, height, {'units': 'm'}),
            'deposition_date': (dims, dep),
            'grain_type': (dims, gtype, {'units': 'Swiss Code F1F2F3'}),
            'sk38': (dims, sk38, {'units': '1'}),
            'HS': (dims[:-1], np.nanmax(np.nan_to_num(height, nan=0), -1),
                   {'units': 'm'}),
        },
        coords={'location': ['A', 'B'], 'time': times, 'slope': [0, 1],
                'realization': ['NO_EXP'], 'layer': np.arange(K)},
    )
    return xsnowDataset(data)


TAGS = ['2025-11-10', '2025-12-01', '2026-01-15']


def _run_counting(fn):
    started = []
    with Callback(pretask=lambda key, *_: started.append(key)):
        out = fn()
    return out, started


def test_assign_datetags_parity_and_laziness(season):
    eager = season.assign_datetags(TAGS)
    lazy_in = xsnowDataset(season.data.chunk({'location': 1, 'layer': 10}))
    lazy, started = _run_counting(lambda: lazy_in.assign_datetags(TAGS))
    assert started == []
    tag = lazy.data['datetag']
    assert isinstance(tag.data, da.Array)
    assert tag.chunks[-1] == (season.sizes['layer'],)
    assert tag.dims == eager.data['datetag'].dims
    assert tag.attrs[DATETAGS_ATTR] == TAGS
    np.testing.assert_array_equal(
        tag.compute().values, eager.data['datetag'].values
    )
    # every tagged layer carries the first tag at/after its deposition day
    dep = season.data['deposition_date'].values.astype('datetime64[D]')
    got = eager.data['datetag'].values.astype('datetime64[D]')
    tags = np.array(TAGS, dtype='datetime64[D]')
    finite = ~np.isnat(dep)
    expect = np.full(dep.shape, np.datetime64('NaT', 'D'))
    idx = np.searchsorted(tags, dep[finite])
    inside = idx < tags.size
    vals = np.full(idx.shape, np.datetime64('NaT', 'D'))
    vals[inside] = tags[idx[inside]]
    expect[finite] = vals
    np.testing.assert_array_equal(got, expect)


def test_assign_datetags_per_location_mapping(season):
    mapping = {
        'meta': {'level1': 'location', 'level2': 'location'},
        'A': {'A': TAGS[:2]},
        'B': {'B': TAGS[1:]},
    }
    eager = season.assign_datetags(mapping)
    lazy_in = xsnowDataset(season.data.chunk({'location': 1}))
    lazy = lazy_in.assign_datetags(mapping)
    np.testing.assert_array_equal(
        lazy.data['datetag'].compute().values, eager.data['datetag'].values
    )
    a = eager.data['datetag'].sel(location='A').values.astype('datetime64[D]')
    assert np.datetime64(TAGS[2], 'D') not in a[~np.isnat(a)]
    assert lazy.data['datetag'].attrs[DATETAGS_ATTR] == TAGS


def test_simplified_stratigraphy_parity_and_laziness(season):
    tagged = season.assign_datetags(TAGS)
    eager = tagged.compute_simplified_stratigraphy()
    lazy_in = xsnowDataset(tagged.data.chunk({'location': 1, 'time': 16,
                                              'layer': 10}))
    lazy, started = _run_counting(
        lambda: lazy_in.compute_simplified_stratigraphy()
    )
    assert started == []
    for name in ('sk38', 'sk38_depth', 'sk38_gcode_weak',
                 'sk38_gcode_deep', 'datetag_layer_count'):
        assert isinstance(lazy.data[name].data, da.Array), name
        assert lazy.data[name].dims == eager.data[name].dims
        assert lazy.data[name].chunks[-1] == (eager.sizes['layer'],)
    assert list(lazy.data['datetag'].values) == TAGS + [
        'shallow_old_snow', 'new_snow'
    ]
    xr.testing.assert_identical(
        lazy.data.compute().drop_vars('datetag'),
        eager.data.drop_vars('datetag'),
    )
    assert 'HS' in lazy.data and 'layer' not in lazy.data['HS'].dims
    # groups overlap (new snow is counted under its tag as well), so the
    # group counts cover every layer at least once
    counts = eager.data['datetag_layer_count'].sum('layer')
    n_layers = np.isfinite(season.data['height']).sum('layer')
    assert (counts.values >= n_layers.values).all()
    tagged_only = eager.data['datetag_layer_count'].isel(
        layer=slice(0, len(TAGS) + 1)
    ).sum('layer')
    assert (tagged_only.values <= n_layers.values).all()


def test_group_list_from_attribute_keeps_absent_tags(season):
    # a tag before the first deposition date tags no layer at all
    tagged = season.assign_datetags(['2025-10-01'] + TAGS)
    red = tagged.compute_simplified_stratigraphy()
    labels = list(red.data['datetag'].values)
    assert labels[0] == '2025-10-01'
    assert int(red.data['datetag_layer_count'].isel(layer=0).sum()) == 0


def _lazy_reduction_on_archive(path, n_locations):
    xs = xsnow.read(path, lazy=True)
    xs = xsnow.xsnowDataset(xs.data.isel(location=slice(0, n_locations)))
    tags = ['1995-12-01', '1996-01-15', '1996-02-15', '1996-03-15']
    started = []
    with Callback(pretask=lambda key, *_: started.append(key)):
        red = xs.assign_datetags(tags).compute_simplified_stratigraphy(
            indices=['sk38', 'rc_flat']
        )
    assert started == []
    assert isinstance(red.data['sk38'].data, da.Array)
    assert red.sizes['layer'] == len(tags) + 2
    one = red.data[['sk38', 'sk38_depth', 'datetag_layer_count']].isel(
        location=0, slope=0, realization=0
    ).compute()
    assert int(one['datetag_layer_count'].max()) > 0
    return red, one


@pytest.mark.skipif(not STATION_ARCHIVE.exists(), reason='no real archive')
def test_reduction_on_real_station_archive():
    red, one = _lazy_reduction_on_archive(STATION_ARCHIVE, 1)
    assert red.sizes['slope'] == 5
    assert np.nanmax(one['sk38_depth'].values) > 0.5


@pytest.mark.skipif(not DOMAIN_ARCHIVE.exists(), reason='no domain archive')
def test_reduction_on_domain_archive():
    red, one = _lazy_reduction_on_archive(DOMAIN_ARCHIVE, 8)
    assert red.sizes['location'] == 8
    # whole-domain reduction of two indices stays a bounded, lazy graph
    n_tasks = len(red.data['sk38'].data.dask)
    assert n_tasks < 20_000
