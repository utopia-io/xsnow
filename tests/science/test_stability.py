"""
Unit tests for the critical crack length extension.

This test suite verifies the functionality of the analysis functions,
ensuring they work correctly with the layer-based coordinate system and are
hardware-aware.
"""
import pytest
import xarray as xr
import numpy as np
import pandas as pd

from xsnow import xsnowDataset
from xsnow.extensions import critical_crack_length  # all methods are now registered on xsnowDataset

@pytest.fixture
def analysis_ds():
    """Provides a sample xsnowDataset suitable for analysis functions."""
    data = {
        'height': (('location', 'time', 'layer'), np.array([[[0.1, 0.2, 0.3, 0.4]]])),
        'density': (('location', 'time', 'layer'), np.array([[[150, 250, 350, 450]]])),
        'grain_size': (('location', 'time', 'layer'), np.array([[[0.0002, 0.0002, 0.0005, 0.0010]]])),
        'shear_strength': (('location', 'time', 'layer'), np.array([[[100.0, 200.0, 800.0, 1500.0]]]))
    }
    coords = {
        'location': [0],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': [0, 1, 2, 3],
    }
    xr_ds = xr.Dataset(data, coords)
    return xsnowDataset(xr_ds)

def test_compute_critical_crack_length(analysis_ds):
    """Tests the calculation of critical cut length."""
    # FIX: Call as a method on the dataset object
    ds_with_rc = analysis_ds.compute_critical_crack_length()
    ds_with_rc_cpu = ds_with_rc.to_cpu()

    assert 'rc_flat' in ds_with_rc_cpu
    assert ds_with_rc_cpu['rc_flat'].dims == ('location', 'time', 'layer')

    surface_val = ds_with_rc_cpu['rc_flat'].isel(layer=-1).item()
    assert np.isnan(surface_val)

    other_vals = ds_with_rc_cpu['rc_flat'].isel(layer=slice(None, -1)).values
    assert np.all(other_vals > 0)
    assert np.nanmax(other_vals) < 100
    assert ds_with_rc_cpu['rc_flat'].attrs['units'] == 'm'
