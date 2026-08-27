"""
Unit tests for the rta extension.

This test suite verifies the functionality of the structural
instability index, ensuring it works correctly with the layer-based
coordinate system and reproduces the verified reference values.
"""
import pytest
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
from dask.callbacks import Callback

from xsnow import xsnowDataset
from xsnow.extensions.rta import RTA_THRESHOLD
from xsnow.extensions import rta  # noqa: F401 -- registers the methods


@pytest.fixture
def analysis_ds():
    """Provides a sample xsnowDataset with a weak layer at index 2."""
    n = 10
    height = 0.12 * np.arange(1, n + 1)
    hardness = np.linspace(4.0, 1.5, n)
    gsize = np.full(n, 0.0005)
    gtype = np.full(n, 330.0)
    hardness[2], gsize[2], gtype[2] = 1.0, 0.0015, 430.0
    data = {
        var: (('location', 'time', 'layer'), arr[None, None, :])
        for var, arr in (('height', height), ('hand_hardness', hardness),
                         ('grain_size', gsize), ('grain_type', gtype))
    }
    coords = {
        'location': [0],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': np.arange(n),
    }
    return xsnowDataset(xr.Dataset(data, coords))


def test_compute_rta(analysis_ds):
    """Tests the calculation of the structural instability index."""
    ds = analysis_ds.compute_rta()

    assert 'rta' in ds.data
    assert ds['rta'].dims == ('location', 'time', 'layer')
    assert ds['rta'].attrs['units'] == '1'

    vals = ds['rta'].values[0, 0]
    assert np.all((vals >= 0) & (vals <= 1))
    assert np.nanargmax(vals) == 2             # the weak layer

    expected = np.array([0.0005, 0.4142, 1.0, 0.4692, 0.2063,
                         0.2786, 0.3027, 0.2663, 0.2007, 0.1622])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_float32_storage_stays_float32(analysis_ds):
    """Computation follows the storage dtype (no float64 upcasts)."""
    f32 = xsnowDataset(analysis_ds.data.astype(np.float32))
    assert f32.compute_rta()['rta'].dtype == np.float32


def test_compute_rta_remains_lazy(analysis_ds):
    expected = analysis_ds.compute_rta()['rta']
    lazy = xsnowDataset(analysis_ds.data.chunk({'layer': 3}))
    started_tasks = []
    with Callback(pretask=lambda key, *_: started_tasks.append(key)):
        actual = lazy.compute_rta()['rta']

    assert started_tasks == []
    assert isinstance(actual.data, da.Array)
    assert actual.chunks[-1] == (actual.sizes['layer'],)
    xr.testing.assert_allclose(actual.compute(), expected)


def test_single_layer_profile_is_zero(analysis_ds):
    """A profile with fewer than two layers has no relative structure."""
    single = xsnowDataset(analysis_ds.data.isel(layer=[0]))
    assert single.compute_rta()['rta'].values[0, 0, 0] == 0.0


def test_missing_variable_raises(analysis_ds):
    broken = xsnowDataset(analysis_ds.data.drop_vars('hand_hardness'))
    with pytest.raises(ValueError, match='hand_hardness'):
        broken.compute_rta()


def _fpel2_profile_0():
    """Observed FPEL2 profile (20 layers) + its computeRTA reference."""
    height_cm = np.array([
        2.43, 4.48, 6.67, 8.73, 10.03, 11.55,
        13.330001, 13.92, 16.44, 17.7, 19.36, 20.57,
        21.45, 23.33, 25.639999, 27.680001, 29.33, 30.8,
        32.260001, 34.110001])
    hardness = np.array([
        5.0, 5.0, 5.0, 5.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 5.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0])
    gsize_mm = np.array([
        1.6, 1.69, 1.76, 1.96, 2.24, 2.14,
        2.24, 2.4, 2.17, 2.05, 1.99, 1.9,
        1.68, 2.18, 1.69, 2.18, 2.33, 0.95,
        1.07, 1.14])
    gtype = np.array([
        772.0, 752.0, 752.0, 752.0, 550.0, 550.0,
        550.0, 550.0, 550.0, 550.0, 550.0, 550.0,
        591.0, 752.0, 550.0, 550.0, 550.0, 490.0,
        490.0, 450.0])
    r_rta = np.array([
        0.275432702827, 0.304337493763, 0.310363187828, 0.486203824227,
        1.0, 0.743661493766, 0.760635567971, 0.791514441583,
        0.733743047296, 0.703363049176, 0.693677139748, 0.696767954026,
        0.844165753117, 0.502575993728, 0.823439550734, 0.710958412655,
        0.915677229767, 0.617284783477, 0.534780033124, 0.431267483698])
    return height_cm, hardness, gsize_mm, gtype, r_rta


def _fpel2_profile_1():
    """Observed FPEL2 profile (32 layers) + its computeRTA reference."""
    height_cm = np.array([
        2.42, 4.44, 6.6, 8.61, 11.21, 12.83,
        13.36, 15.620001, 18.14, 19.19, 19.98, 21.48,
        23.42, 25.940001, 26.989999, 27.959999, 29.100001, 31.830001,
        34.27, 36.08, 38.119999, 39.050001, 41.839999, 43.529999,
        45.24, 48.210001, 49.939999, 51.880002, 53.750002, 55.830002,
        57.999998, 60.339999])
    hardness = np.array([
        5.0, 5.0, 5.0, 5.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.5, 5.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.5, 1.0, 1.0, 1.5, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0])
    gsize_mm = np.array([
        1.7, 1.79, 1.86, 2.06, 2.31, 2.37,
        2.52, 2.31, 2.19, 2.09, 1.84, 2.37,
        1.91, 2.46, 1.34, 1.47, 1.56, 1.27,
        0.77, 0.66, 0.72, 0.76, 0.65, 0.74,
        0.76, 0.87, 1.29, 0.44, 0.4, 0.41,
        0.5, 0.3])
    gtype = np.array([
        752.0, 752.0, 752.0, 752.0, 550.0, 550.0,
        550.0, 550.0, 550.0, 550.0, 591.0, 752.0,
        550.0, 550.0, 450.0, 450.0, 550.0, 491.0,
        490.0, 431.0, 490.0, 490.0, 430.0, 990.0,
        990.0, 490.0, 450.0, 241.0, 430.0, 210.0,
        211.0, 110.0])
    r_rta = np.array([
        0.314830204295, 0.323161970986, 0.356129935598, 0.559370774949,
        0.982921514979, 0.791014356354, 0.815190000463, 0.792235016629,
        0.76949429342, 0.808867053274, 0.941752811062, 0.637036802195,
        1.0, 0.970755537071, 0.861923125739, 0.700784211178,
        0.742322391121, 0.749526547771, 0.668954958482, 0.607899031171,
        0.60793570394, 0.643321890958, 0.599328324481, 0.593648241321,
        0.607191091572, 0.664496006535, 0.773127646035, 0.388453330185,
        0.540105480582, 0.356788642918, 0.381259206586, 0.228270412643])
    return height_cm, hardness, gsize_mm, gtype, r_rta


def test_r_reference_classification():
    """The weak-layer classification matches sarp.snowprofile::computeRTA.

    Two observed FPEL2 profiles (hand hardness converted from N to index
    steps); reference computed with sarp.snowprofile 1.4.1 in R on
    2026-08-17. The values differ by up to 0.13 because computeRTA.R
    leaves the below-interface hardness anomaly unstandardized and scales
    the above-interface depth property twice, neither of which this
    implementation follows. Both profiles still agree on the weakest
    layer and on every layer's weak-layer flag.
    """
    for profile in (_fpel2_profile_0(), _fpel2_profile_1()):
        height_cm, hardness, gsize_mm, gtype, r_rta = profile
        n = len(height_cm)
        data = {
            var: (('location', 'time', 'layer'), arr[None, None, :])
            for var, arr in (('height', height_cm * 0.01),
                             ('hand_hardness', hardness),
                             ('grain_size', gsize_mm * 0.001),
                             ('grain_type', gtype))
        }
        coords = {
            'location': [0],
            'time': [pd.to_datetime('2024-01-01')],
            'layer': np.arange(n),
        }
        ds = xsnowDataset(xr.Dataset(data, coords))
        vals = ds.compute_rta()['rta'].values[0, 0]
        assert np.nanargmax(vals) == np.nanargmax(r_rta)
        np.testing.assert_array_equal(vals >= RTA_THRESHOLD,
                                      r_rta >= RTA_THRESHOLD)
        np.testing.assert_allclose(vals, r_rta, atol=0.15)
