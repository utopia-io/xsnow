# tests/test_analysis.py
"""
Unit tests for the analysis module.

This test suite verifies the functionality of the analysis functions,
ensuring they work correctly with the layer-based coordinate system and are
hardware-aware.
"""

import pytest
import xarray as xr
import numpy as np
import pandas as pd

from xsnow.dataset import xsnowDataset
from xsnow.analysis import find_layer_by_criteria, compute_critical_cut_length

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
        'location': ['STATION_A'],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': [0, 1, 2, 3],
    }
    xr_ds = xr.Dataset(data, coords)
    return xsnowDataset(xr_ds)

def test_find_layer_by_criteria(analysis_ds):
    """Tests filtering layers based on a string criterion."""
    # Find layers where density is less than 300
    criteria = "density < 300"
    filtered_ds = find_layer_by_criteria(analysis_ds, criteria)

    # The last two layers (350, 450) should be NaN
    expected = np.array([[[150, 250, np.nan, np.nan]]])
    actual = filtered_ds.data['density'].values

    np.testing.assert_allclose(expected, actual, equal_nan=True)

def test_compute_critical_cut_length(analysis_ds):
    """Tests the calculation of critical cut length."""
    # This might return a GPU-backed dataset if cupy is installed
    ds_with_rc = compute_critical_cut_length(analysis_ds)

    # FIX: Explicitly move the result to the CPU before making assertions
    ds_with_rc_cpu = ds_with_rc.to_cpu()

    assert 'rc_flat' in ds_with_rc_cpu.data
    assert ds_with_rc_cpu.data['rc_flat'].dims == ('location', 'time', 'layer')

    # Check that the surface layer (max layer index) is NaN
    # This now works because the data is guaranteed to be on the CPU
    surface_val = ds_with_rc_cpu.data['rc_flat'].isel(layer=-1).item()
    assert np.isnan(surface_val)

    # Check that other values are positive numbers
    # This also works now
    other_vals = ds_with_rc_cpu.data['rc_flat'].isel(layer=slice(None, -1)).values
    assert np.all(other_vals > 0)