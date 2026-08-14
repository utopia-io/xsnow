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

from xsnow import xsnowDataset
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

    # values of the implementation verified bit-exactly against
    # sarp.snowprofile::computeRTA (2026-08-14)
    expected = np.array([0.0804, 0.4817, 1.0, 0.4597, 0.2021,
                         0.2730, 0.2966, 0.2609, 0.2421, 0.1589])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_single_layer_profile_is_zero(analysis_ds):
    """A profile with fewer than two layers has no relative structure."""
    single = xsnowDataset(analysis_ds.data.isel(layer=[0]))
    assert single.compute_rta()['rta'].values[0, 0, 0] == 0.0


def test_missing_variable_raises(analysis_ds):
    broken = xsnowDataset(analysis_ds.data.drop_vars('hand_hardness'))
    with pytest.raises(ValueError, match='hand_hardness'):
        broken.compute_rta()
