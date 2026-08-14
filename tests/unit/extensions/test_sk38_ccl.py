"""
Unit tests for the sk38 and sk38_ccl extensions.

This test suite verifies the functionality of the skier stability
indices, ensuring they work correctly with the layer-based coordinate
system and reproduce the verified reference values (implementation
verified against the R/operational reference outputs, 2026-08-14).
"""
import pytest
import xarray as xr
import numpy as np
import pandas as pd

from xsnow import xsnowDataset
# importing the module registers all methods on xsnowDataset
from xsnow.extensions.sk38_ccl import INDEX_CAP
from xsnow.extensions.sk38 import compute_skier_penetration


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


def test_compute_sk38_ccl_rta(analysis_ds):
    """Tests the RTA gate and its fallback to compute_rta."""
    gated = analysis_ds.compute_sk38_ccl_rta()
    assert 'sk38_ccl_rta' in gated.data
    assert 'sk38_ccl' not in gated.data

    # gating: every computable layer below the threshold becomes the cap
    low_rta = analysis_ds.data.assign(rta=analysis_ds.data['rta'] * 0.5)
    vals = xsnowDataset(low_rta).compute_sk38_ccl_rta()[
        'sk38_ccl_rta'].values[0, 0]
    finite = vals[np.isfinite(vals)]
    assert finite.size and np.all(finite == INDEX_CAP)

    # without an rta variable the gate falls back to compute_rta
    no_rta = xsnowDataset(analysis_ds.data.drop_vars('rta'))
    assert 'sk38_ccl_rta' in no_rta.compute_sk38_ccl_rta().data


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


def test_missing_variable_raises(analysis_ds):
    broken = xsnowDataset(analysis_ds.data.drop_vars('shear_strength'))
    with pytest.raises(ValueError, match='shear_strength'):
        broken.compute_sk38_ccl()


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
        monkeypatch.setattr(mod, 'get_backend', lambda ds: CupySurface())

    assert 'sk38' in analysis_ds.compute_sk38().data
    assert 'sk38_ccl_rta' in analysis_ds.compute_sk38_ccl_rta().data
    assert 'rta' in analysis_ds.compute_rta().data
    assert 'rc_slope' in analysis_ds.compute_critical_crack_length(
        inclination=38.0).data
