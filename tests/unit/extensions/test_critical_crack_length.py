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

    # values of the verified implementation (2026-08-17); NaN = topmost
    expected = np.array([3.8425, 4.2956, 3.1305, np.nan])
    np.testing.assert_allclose(
        ds_with_rc_cpu['rc_flat'].values[0, 0], expected, atol=1e-3
    )


def test_compute_critical_crack_length_slope(analysis_ds):
    """Tests the slope formulation of the critical crack length."""
    strong = xsnowDataset(analysis_ds.data.assign(
        shear_strength=analysis_ds.data['shear_strength'] * 5,
        grain_type=(analysis_ds.data['height'].dims,
                    np.array([[[430.0, 330.0, 330.0, 330.0]]])),
    ))
    ds_rc = strong.compute_critical_crack_length(inclination=38.0)

    assert 'rc_slope' in ds_rc.data
    vals = ds_rc['rc_slope'].values[0, 0]
    # values of the verified implementation (2026-08-14); NaN = topmost
    expected = np.array([0.5286, 2.7814, 4.4009, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)

    # the slope formulation additionally requires the grain type
    with pytest.raises(ValueError, match='grain_type'):
        analysis_ds.compute_critical_crack_length(inclination=38.0)

    # without the strengthening the two lowest layers are in the
    # spontaneous regime (slope-parallel load exceeds the corrected
    # strength) and rc_slope is NaN there
    weak = xsnowDataset(analysis_ds.data.assign(
        grain_type=(analysis_ds.data['height'].dims,
                    np.array([[[430.0, 330.0, 330.0, 330.0]]])),
    ))
    vals = weak.compute_critical_crack_length(
        inclination=38.0)['rc_slope'].values[0, 0]
    expected = np.array([np.nan, np.nan, 1.4774, np.nan])
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_interior_nan_height_raises(analysis_ds):
    """A NaN height below the surface breaks the layer contiguity the
    slab sums rely on, so it is rejected; NaN at the top is padding.
    Index 2 is the gap directly under the surface layer, whose derived
    thicknesses look exactly like padding."""
    grain_type = (analysis_ds.data['height'].dims,
                  np.array([[[430.0, 330.0, 330.0, 330.0]]]))
    for idx in (1, 2):
        height = analysis_ds.data['height'].copy()
        height[0, 0, idx] = np.nan
        broken_xr = analysis_ds.data.assign(height=height)
        with pytest.raises(ValueError, match='interior'):
            xsnowDataset(broken_xr).compute_critical_crack_length()
        # the slope path guards through the shared slab mechanics
        with_gt = xsnowDataset(broken_xr.assign(grain_type=grain_type))
        with pytest.raises(ValueError, match='interior'):
            with_gt.compute_critical_crack_length(inclination=38.0)

    padded = analysis_ds.data['height'].copy()
    padded[0, 0, -1] = np.nan
    ok = xsnowDataset(analysis_ds.data.assign(height=padded))
    assert 'rc_flat' in ok.compute_critical_crack_length().data
