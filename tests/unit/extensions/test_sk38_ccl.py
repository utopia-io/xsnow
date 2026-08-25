"""
Unit tests for the sk38 and sk38_ccl extensions.

This test suite verifies the functionality of the skier stability
indices, ensuring they work correctly with the layer-based coordinate
system and reproduce the verified reference values (implementation
verified against the R/operational reference outputs, 2026-08-14).
"""
from dataclasses import fields

import pytest
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
from dask.callbacks import Callback

from xsnow import xsnowDataset
# importing the module registers all methods on xsnowDataset
from xsnow.extensions.critical_crack_length import compute_slab_mechanics
from xsnow.extensions.sk38_ccl import INDEX_CAP
from xsnow.extensions.sk38 import (
    _compute_skier_mechanics,
    compute_skier_penetration,
)


@pytest.fixture
def analysis_ds():
    """Provides a sample xsnowDataset with a weak layer at index 2."""
    n = 8
    height = 0.12 * np.arange(1, n + 1)
    density = np.linspace(320.0, 120.0, n)
    shear = np.linspace(4200.0, 900.0, n)
    gsize = np.full(n, 0.0005)
    gtype = np.full(n, 330.0)
    hardness = np.linspace(4.0, 1.5, n)
    rta = np.full(n, 0.9)
    density[2], shear[2], gsize[2], gtype[2] = 140.0, 300.0, 0.0015, 430.0
    hardness[2] = 1.0
    data = {
        var: (('location', 'time', 'layer'), arr[None, None, :])
        for var, arr in (('height', height), ('density', density),
                         ('shear_strength', shear), ('grain_size', gsize),
                         ('grain_type', gtype), ('hand_hardness', hardness),
                         ('rta', rta))
    }
    coords = {
        'location': [0],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': np.arange(n),
    }
    return xsnowDataset(xr.Dataset(data, coords))


def test_compute_sk38(analysis_ds):
    """Tests the calculation of the skier stability index SK38."""
    ds = analysis_ds.compute_sk38()

    assert 'sk38' in ds.data
    assert ds['sk38'].dims == ('location', 'time', 'layer')
    assert ds['sk38'].attrs['units'] == '1'

    vals = ds['sk38'].values[0, 0]
    # NaN above: skier-penetrated layers and the topmost layer
    expected = np.array([2.3533, 1.8629, 0.4389, 1.1505, 0.5623,
                         np.nan, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_compute_sk38_ccl(analysis_ds):
    """Tests the calculation of the combined index sk38_ccl."""
    ds = analysis_ds.compute_sk38_ccl()

    assert 'sk38_ccl' in ds.data
    assert ds['sk38_ccl'].attrs['index_cap'] == INDEX_CAP

    vals = ds['sk38_ccl'].values[0, 0]
    # 0 = spontaneous weak layer, INDEX_CAP = stable, NaN = penetrated/top
    expected = np.array([5.0, 5.0, 0.0, 5.0, 5.0, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)

    # the index owns .pro parameter code 0608
    from xsnow.io.base import CODE_TO_NAME_MAP
    assert CODE_TO_NAME_MAP['0608'] == 'sk38_ccl'


def test_mechanics_records_retain_only_reusable_results(analysis_ds):
    """Block-sized intermediates do not outlive mechanics construction."""
    density = analysis_ds['density'].values
    shear = analysis_ds['shear_strength'].values
    height = analysis_ds['height'].values
    grain_type = analysis_ds['grain_type'].values
    thickness = np.diff(
        height,
        axis=-1,
        prepend=np.zeros_like(height[..., :1]),
    )

    slab = compute_slab_mechanics(
        density, thickness, height, grain_type, shear, 38.0, np
    )
    assert tuple(field.name for field in fields(slab)) == (
        'valid', 'density', 'height_n', 'thick_n', 'surface_n',
        'e_layer', 'h_equiv', 'tau_g', 'sigma_n', 'tau_p',
    )
    assert not hasattr(slab, '__dict__')

    skier = _compute_skier_mechanics(
        density, shear, height, grain_type, np
    )
    assert tuple(field.name for field in fields(skier)) == (
        'valid', 'density', 'h_equiv', 'tau_g', 'sigma_n', 'tau_p',
        'inclination_rad', 'line_load', 'penetrated', 'h_equiv_sk',
        'top_layer',
    )
    assert not hasattr(skier, '__dict__')


def test_compute_sk38_ccl_soft_slab(analysis_ds):
    """A weak snowpack yields uncapped indices: the crack-length ratio
    itself, not the cap or a sentinel, determines the values."""
    soft = xsnowDataset(analysis_ds.data.assign(
        shear_strength=analysis_ds.data['shear_strength'] * 0.3
    ))
    vals = soft.compute_sk38_ccl()['sk38_ccl'].values[0, 0]

    # values of the verified implementation (2026-08-16)
    expected = np.array([3.3764, 2.7902, 0.0, 1.9826, 2.0889,
                         np.nan, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_compute_sk38_ccl_rta(analysis_ds):
    """Tests the RTA gate and its fallback to compute_rta."""
    gated = analysis_ds.compute_sk38_ccl_rta()
    assert 'sk38_ccl_rta' in gated.data
    assert 'sk38_ccl' not in gated.data

    # rta above the threshold everywhere: the index passes unchanged
    expected = np.array([5.0, 5.0, 0.0, 5.0, 5.0, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(
        gated['sk38_ccl_rta'].values[0, 0], expected, atol=1e-3
    )

    # gating: every computable layer below the threshold becomes the cap
    low_rta = analysis_ds.data.assign(rta=analysis_ds.data['rta'] * 0.5)
    vals = xsnowDataset(low_rta).compute_sk38_ccl_rta()[
        'sk38_ccl_rta'].values[0, 0]
    finite = vals[np.isfinite(vals)]
    assert finite.size and np.all(finite == INDEX_CAP)
    assert np.isnan(vals[5:]).all()      # NaN layers stay NaN when gated

    # without an rta variable the gate falls back to compute_rta
    no_rta = xsnowDataset(analysis_ds.data.drop_vars('rta'))
    assert 'sk38_ccl_rta' in no_rta.compute_sk38_ccl_rta().data


@pytest.mark.parametrize(
    'method_name, out_name',
    [
        ('compute_sk38', 'sk38'),
        ('compute_sk38_ccl', 'sk38_ccl'),
        ('compute_sk38_ccl_rta', 'sk38_ccl_rta'),
    ],
)
def test_skier_indices_remain_lazy(analysis_ds, method_name, out_name):
    expected = getattr(analysis_ds, method_name)()[out_name]
    lazy = xsnowDataset(analysis_ds.data.chunk({'layer': 3}))
    started_tasks = []
    with Callback(pretask=lambda key, *_: started_tasks.append(key)):
        actual = getattr(lazy, method_name)()[out_name]

    assert started_tasks == []
    assert isinstance(actual.data, da.Array)
    assert sum(actual.chunks[-1]) == actual.sizes['layer']
    if method_name != 'compute_sk38_ccl_rta':
        assert actual.chunks[-1] == (actual.sizes['layer'],)
    xr.testing.assert_allclose(actual.compute(), expected)


def test_sk38_ccl_rta_fallback_remains_lazy(analysis_ds):
    eager = xsnowDataset(analysis_ds.data.drop_vars('rta'))
    expected = eager.compute_sk38_ccl_rta()
    lazy = xsnowDataset(eager.data.chunk({'layer': 3}))
    started_tasks = []
    with Callback(pretask=lambda key, *_: started_tasks.append(key)):
        actual = lazy.compute_sk38_ccl_rta()

    assert started_tasks == []
    assert isinstance(actual['rta'].data, da.Array)
    assert isinstance(actual['sk38_ccl_rta'].data, da.Array)
    xr.testing.assert_allclose(
        actual[['rta', 'sk38_ccl_rta']].compute(),
        expected[['rta', 'sk38_ccl_rta']],
    )


def test_missing_inputs_are_never_stable(analysis_ds):
    """A layer with missing input data is NaN, not the stable cap."""
    shear = analysis_ds.data['shear_strength'].copy()
    shear[0, 0, 1] = np.nan
    with_nan = xsnowDataset(analysis_ds.data.assign(shear_strength=shear))
    vals = with_nan.compute_sk38_ccl()['sk38_ccl'].values[0, 0]
    expected = np.array([5.0, np.nan, 0.0, 5.0, 5.0,
                         np.nan, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)

    # a missing density leaves the slab of every layer below it undefined
    density = analysis_ds.data['density'].copy()
    density[0, 0, 3] = np.nan
    with_nan = xsnowDataset(analysis_ds.data.assign(density=density))
    vals = with_nan.compute_sk38_ccl()['sk38_ccl'].values[0, 0]
    expected = np.array([np.nan, np.nan, np.nan, np.nan, 5.0,
                         np.nan, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)

    # the strength correction depends on the grain form, so a missing
    # grain type leaves the layer's verdict undefined (here: the weak
    # layer, whose spontaneous 0.0 must not survive without it)
    gtype = analysis_ds.data['grain_type'].copy()
    gtype[0, 0, 2] = np.nan
    with_nan = xsnowDataset(analysis_ds.data.assign(grain_type=gtype))
    vals = with_nan.compute_sk38_ccl()['sk38_ccl'].values[0, 0]
    expected = np.array([5.0, 5.0, np.nan, 5.0, 5.0,
                         np.nan, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_spontaneous_penetrated_layer_stays_zero():
    """A spontaneous layer inside the skier penetration keeps its 0.0:
    the verdict needs only the strength excess, not the equivalent slab
    thickness (undefined for the first penetrated layer)."""
    n = 8
    height = 0.12 * np.arange(1, n + 1)
    shear = np.full(n, 5000.0)
    shear[5] = 1.0
    data = {
        var: (('location', 'time', 'layer'), arr[None, None, :])
        for var, arr in (('height', height),
                         ('density', np.full(n, 100.0)),
                         ('shear_strength', shear),
                         ('grain_size', np.full(n, 0.0005)),
                         ('grain_type', np.full(n, 330.0)))
    }
    coords = {
        'location': [0],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': np.arange(n),
    }
    ds = xsnowDataset(xr.Dataset(data, coords))
    vals = ds.compute_sk38_ccl()['sk38_ccl'].values[0, 0]
    expected = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 0.0, np.nan, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_float32_storage_stays_float32(analysis_ds):
    """Computation follows the storage dtype (no float64 upcasts)."""
    f32 = xsnowDataset(analysis_ds.data.astype(np.float32))
    assert f32.compute_sk38()['sk38'].dtype == np.float32
    assert f32.compute_sk38_ccl()['sk38_ccl'].dtype == np.float32


def test_skier_penetration_crust_limit():
    """A thick melt-freeze crust stops the skier at its top."""
    # 35 cm of light snow (rho=80) above a 4 cm dense crust:
    # unimpeded penetration would be 34.6/80 = 0.4325 m
    height = np.array([0.10, 0.20, 0.30, 0.34, 0.45, 0.57, 0.69])
    thick = np.diff(np.concatenate([[0.0], height]))
    density = np.array([300.0, 300.0, 300.0, 550.0, 80.0, 80.0, 80.0])
    gtype = np.full(7, 330.0)
    gtype_crust = gtype.copy()
    gtype_crust[3] = 772.0

    pen_soft = compute_skier_penetration(
        density[None, :], thick[None, :], gtype[None, :], np)[0]
    pen_crust = compute_skier_penetration(
        density[None, :], thick[None, :], gtype_crust[None, :], np)[0]
    assert pen_soft == pytest.approx(34.6 / 80.0)
    assert pen_crust == pytest.approx(thick[4:].sum())     # 0.35 m


def test_skier_penetration_excludes_unknown_density():
    """A layer of unknown density does not enter the top-30 cm average."""
    height = np.array([0.4, 0.5])
    thick = np.diff(np.concatenate([[0.0], height]))
    density = np.array([200.0, np.nan])
    gtype = np.full(2, 330.0)
    pen = compute_skier_penetration(
        density[None, :], thick[None, :], gtype[None, :], np)[0]
    assert pen == pytest.approx(34.6 / 200.0)


def test_rta_gate_keeps_existing_sk38_ccl(analysis_ds):
    """A pre-existing sk38_ccl survives the gate (recomputed); the
    internally created intermediate is still dropped (see
    test_compute_sk38_ccl_rta)."""
    chained = analysis_ds.compute_sk38_ccl().compute_sk38_ccl_rta()
    assert 'sk38_ccl' in chained.data
    np.testing.assert_allclose(
        chained['sk38_ccl'].values,
        analysis_ds.compute_sk38_ccl()['sk38_ccl'].values,
    )


def test_missing_variable_raises(analysis_ds):
    broken = xsnowDataset(analysis_ds.data.drop_vars('shear_strength'))
    with pytest.raises(ValueError, match='shear_strength'):
        broken.compute_sk38_ccl()


def _profile_ds(**arrays):
    """In-memory single-profile dataset from per-layer arrays."""
    n = len(arrays['height'])
    data = {var: (('location', 'time', 'layer'), arr[None, None, :])
            for var, arr in arrays.items()}
    coords = {'location': [0], 'time': [pd.to_datetime('2024-01-01')],
              'layer': np.arange(n)}
    return xsnowDataset(xr.Dataset(data, coords))


def _gess2_2023_11_03():
    """GESS2 profile 2023-11-03 07:00 + its R pipeline reference row."""
    height = np.array([
        0.0114, 0.0236, 0.0351, 0.0542, 0.0733, 0.0865,
        0.1064, 0.1233, 0.1396, 0.1566, 0.1715, 0.1865,
        0.2016, 0.2174, 0.2343, 0.251, 0.2676, 0.2843,
        0.2946, 0.3094, 0.3338, 0.3586, 0.3838, 0.4025,
        0.4203, 0.4354, 0.4545, 0.4729, 0.5102, 0.5282,
        0.5474, 0.5678, 0.596, 0.6198, 0.6432, 0.6676,
        0.6927])
    density = np.array([
        392.899994, 369.700012, 271.299988, 266.5, 257.100006, 248.300003,
        238.899994, 228.800003, 220.5, 213.0, 222.800003, 220.699997,
        219.100006, 214.0, 206.699997, 187.199997, 187.300003, 185.600006,
        143.199997, 153.199997, 136.600006, 134.5, 132.0, 129.899994,
        120.599998, 120.800003, 149.0, 145.0, 134.600006, 84.5,
        87.199997, 85.0, 83.800003, 81.099998, 76.400002, 90.400002,
        94.099998])
    shear_strength = np.array([
        4000.0, 6810.0, 1340.0, 1310.0, 1230.0, 1170.0,
        1100.0, 1030.0, 970.0, 920.0, 990.0, 980.0,
        970.0, 1290.0, 1240.0, 1100.0, 1100.0, 1090.0,
        500.0, 380.0, 330.0, 330.0, 320.0, 310.0,
        290.0, 290.0, 530.0, 510.0, 460.0, 190.0,
        200.0, 190.0, 190.0, 180.0, 170.0, 210.0,
        220.0])
    grain_size = np.array([
        0.00094, 0.0009, 0.0003, 0.0003, 0.0003, 0.0003,
        0.00036, 0.00036, 0.00036, 0.00033, 0.00034, 0.00034,
        0.00034, 0.00033, 0.00034, 0.00039, 0.00042, 0.00043,
        0.00056, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.00031, 0.0003, 0.00031, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003])
    grain_type = np.array([
        770.0, 770.0, 220.0, 211.0, 220.0, 220.0,
        220.0, 220.0, 220.0, 220.0, 230.0, 230.0,
        230.0, 321.0, 321.0, 341.0, 341.0, 341.0,
        220.0, 110.0, 110.0, 110.0, 110.0, 110.0,
        110.0, 110.0, 220.0, 220.0, 220.0, 110.0,
        110.0, 110.0, 110.0, 110.0, 110.0, 110.0,
        110.0])
    rta = np.array([
        0.34, 1.0, 0.19, 0.41, 0.3, 0.31,
        0.35, 0.36, 0.37, 0.59, 0.39, 0.39,
        0.4, 0.51, 0.46, 0.86, 0.79, 0.8,
        0.84, 0.42, 0.62, 0.49, 0.48, 0.47,
        0.46, 0.45, 0.44, 0.42, 0.4, 0.37,
        0.35, 0.33, 0.31, 0.29, 0.27, 0.25,
        0.0])
    r_reference = np.array([
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 4.828989, 10000.0, 5.282934,
        0.895113, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0])
    inputs = dict(height=height, density=density,
                  shear_strength=shear_strength, grain_size=grain_size,
                  grain_type=grain_type, rta=rta)
    return inputs, r_reference


def _gess2_2023_12_01():
    """GESS2 profile 2023-12-01 19:00 + its R pipeline reference row."""
    height = np.array([
        0.0223, 0.0439, 0.0572, 0.0804, 0.0919, 0.1136,
        0.1323, 0.1515, 0.1616, 0.18, 0.1893, 0.1943,
        0.2129, 0.2356, 0.2579, 0.2759, 0.2927, 0.3048,
        0.3211, 0.3357, 0.3533, 0.3703, 0.38, 0.4006,
        0.4215, 0.4359, 0.4469, 0.4634, 0.4864, 0.5068,
        0.5297, 0.5512, 0.5776, 0.6076, 0.6306])
    density = np.array([
        404.600006, 388.899994, 376.399994, 353.799988, 345.799988,
        339.899994, 361.399994, 357.5, 352.399994, 345.700012,
        343.700012, 303.0, 309.0, 303.200012, 295.100006,
        316.600006, 309.0, 272.299988, 319.899994, 347.899994,
        326.899994, 298.799988, 268.100006, 145.5, 136.899994,
        119.400002, 146.699997, 136.100006, 109.0, 149.5,
        139.5, 136.600006, 99.300003, 96.900002, 95.699997])
    shear_strength = np.array([
        4000.0, 3030.0, 2830.0, 2480.0, 2360.0, 2280.0,
        2590.0, 2530.0, 2460.0, 2360.0, 2330.0, 1790.0,
        1860.0, 1790.0, 1690.0, 1960.0, 1860.0, 1430.0,
        4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 510.0,
        330.0, 250.0, 520.0, 330.0, 210.0, 540.0,
        480.0, 470.0, 230.0, 220.0, 220.0])
    grain_size = np.array([
        0.00112, 0.00074, 0.00076, 0.00083, 0.00084, 0.00083,
        0.00085, 0.00085, 0.00087, 0.0009, 0.00091, 0.00095,
        0.00084, 0.00084, 0.00086, 0.00089, 0.00092, 0.00096,
        0.00086, 0.00082, 0.00091, 0.001, 0.00112, 0.00066,
        0.00067, 0.00084, 0.0005, 0.00047, 0.00044, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003])
    grain_type = np.array([
        772.0, 990.0, 990.0, 490.0, 490.0, 490.0,
        990.0, 990.0, 990.0, 990.0, 990.0, 490.0,
        490.0, 490.0, 490.0, 990.0, 990.0, 490.0,
        772.0, 772.0, 772.0, 772.0, 792.0, 241.0,
        421.0, 421.0, 241.0, 421.0, 421.0, 220.0,
        220.0, 220.0, 110.0, 110.0, 110.0])
    rta = np.array([
        0.28, 0.8, 0.54, 0.57, 0.59, 0.6,
        0.62, 0.63, 0.65, 0.66, 0.67, 0.81,
        0.71, 0.71, 0.72, 0.72, 0.73, 0.93,
        0.33, 0.32, 0.34, 0.36, 0.5, 1.0,
        0.56, 0.59, 0.49, 0.47, 0.45, 0.28,
        0.27, 0.25, 0.23, 0.21, 0.0])
    r_reference = np.array([
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 7.74896,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 4.182408,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 2.221101,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
    inputs = dict(height=height, density=density,
                  shear_strength=shear_strength, grain_size=grain_size,
                  grain_type=grain_type, rta=rta)
    return inputs, r_reference


def test_reference_gess2():
    """sk38_ccl_rta reproduces the R pipeline reference output.

    Two operational SNOWPACK profiles (GESS2, season 2023/24), hardcoded
    from the full-season comparison verified on 2026-08-14 (256k cells).
    The reference (GESS2.txt) maps stable, RTA-gated and not-computable
    layers all to 10000 and keeps the index uncapped otherwise.
    """
    for inputs, r_reference in (_gess2_2023_11_03(), _gess2_2023_12_01()):
        gated = _profile_ds(**inputs).compute_sk38_ccl_rta()
        vals = gated['sk38_ccl_rta'].values[0, 0]
        assert vals.shape == r_reference.shape
        for k, (rv, mv) in enumerate(zip(r_reference, vals)):
            if rv == 10000.0:
                assert np.isnan(mv) or mv == INDEX_CAP, (k, rv, mv)
            elif rv > INDEX_CAP:
                assert mv == INDEX_CAP, (k, rv, mv)
            else:
                assert np.isfinite(mv), (k, rv, mv)
                assert abs(mv - rv) <= max(1e-3 * abs(rv), 2e-6), (k, rv, mv)


def test_backend_stays_within_cupy_api(analysis_ds, monkeypatch):
    """Execute on numpy restricted to the cupy API surface (no GPU
    required; runs only where cupy is installed)."""
    cupy = pytest.importorskip('cupy')
    assert hasattr(cupy.maximum, 'accumulate')

    class CupySurface:
        def __getattr__(self, name):
            if not hasattr(cupy, name):
                raise AttributeError(f"cupy has no attribute {name!r}")
            return getattr(np, name)

    from xsnow.extensions import (critical_crack_length, rta, sk38,
                                  sk38_ccl)
    for mod in (sk38, sk38_ccl, rta, critical_crack_length):
        monkeypatch.setattr(
            mod, 'get_array_backend', lambda array: CupySurface()
        )

    assert 'sk38' in analysis_ds.compute_sk38().data
    assert 'sk38_ccl_rta' in analysis_ds.compute_sk38_ccl_rta().data
    assert 'rta' in analysis_ds.compute_rta().data
    assert 'rc_flat' in analysis_ds.compute_critical_crack_length().data
    assert 'rc_slope' in analysis_ds.compute_critical_crack_length(
        inclination=38.0).data
