"""
Unit tests for the caic extension.

This test suite verifies the functionality of the analysis functions,
ensuring they work correctly with the layer-based coordinate system and are
hardware-aware.
"""
import pytest
import xarray as xr
import numpy as np
import pandas as pd

from xsnow import xsnowDataset
from xsnow.extensions import caic  # all methods are now registered on xsnowDataset

@pytest.fixture
def analysis_ds():
    """Provides a sample xsnowDataset suitable for analysis functions."""
    data = {
        'height': (('location', 'time', 'layer'), np.array([[[0.1, 0.2, 0.3, 0.4]]])),
        'density': (('location', 'time', 'layer'), np.array([[[150, 250, 350, 450]]])),
        'grain_size': (('location', 'time', 'layer'), np.array([[[0.2, 0.2, 0.5, 1.0]]])),
        'shear_strength': (('location', 'time', 'layer'), np.array([[[0.1, 0.2, 0.8, 1.5]]]))
    }
    coords = {
        'location': [0],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': [0, 1, 2, 3],
    }
    xr_ds = xr.Dataset(data, coords)
    return xsnowDataset(xr_ds)

def test_find_layer_by_criteria(analysis_ds):
    """Tests filtering layers based on a string criterion."""
    criteria = "density < 300"
    # FIX: Call as a method on the dataset object
    filtered_ds = analysis_ds.find_layer_by_criteria(criteria)

    expected = np.array([[[150, 250, np.nan, np.nan]]])
    actual = filtered_ds['density'].values
    np.testing.assert_allclose(expected, actual, equal_nan=True)

def test_compute_critical_cut_length(analysis_ds):
    """Tests the calculation of critical cut length."""
    # FIX: Call as a method on the dataset object
    ds_with_rc = analysis_ds.compute_critical_cut_length()
    ds_with_rc_cpu = ds_with_rc.to_cpu()

    assert 'rc_flat' in ds_with_rc_cpu
    assert ds_with_rc_cpu['rc_flat'].dims == ('location', 'time', 'layer')

    surface_val = ds_with_rc_cpu['rc_flat'].isel(layer=-1).item()
    assert np.isnan(surface_val)

    other_vals = ds_with_rc_cpu['rc_flat'].isel(layer=slice(None, -1)).values
    assert np.all(other_vals > 0)