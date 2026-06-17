# tests/test_dataset.py
"""
Unit tests for the dataset module.

This test suite verifies the functionality of the `xsnowDataset` class,
ensuring that it correctly wraps an xarray.Dataset and that its methods
for copying, analysis, and I/O perform as expected.
"""

import pytest
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import time

from xsnow import xsnowDataset, single_profile_timeseries

# --- Fixtures ---

@pytest.fixture
def sample_xr_dataset():
    """Creates a sample, in-memory xarray.Dataset for testing."""
    layer_coords = [0, 1, 2]
    data = {
        'height': (('location', 'time', 'layer'), np.array([[[0.1, 0.2, 0.3]]])),
        'density': (('location', 'time', 'layer'), np.array([[[100, 200, 300]]])),
        'profile_status': (('location', 'time'), np.array([[1]], dtype='i1')),
        'HS': (('location', 'time'), np.array([[0.3]])),
    }
    coords = {
        'location': ['STATION_A'],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': layer_coords,
        'latitude': ('location', [45.0]),
        'longitude': ('location', [10.0])
    }
    return xr.Dataset(data, coords)

@pytest.fixture
def sample_xsnow_dataset(sample_xr_dataset):
    """Creates a sample xsnowDataset instance from the sample xarray.Dataset."""
    return xsnowDataset(sample_xr_dataset)

@pytest.fixture
def data_dir_for_io_tests(tmp_path):
    """Creates a temporary directory with files for testing I/O methods."""
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    update_dir = tmp_path / "update"
    update_dir.mkdir()

    # Initial file for the .append_latest() test
    (initial_dir / "initial.pro").write_text("""
[STATION_PARAMETERS]
StationName= IO_STATION
[DATA]
0500,09.02.2025 12:00:00
0502,1,90
""")

    # New file for the .append_latest() test
    (update_dir / "update.pro").write_text("""
[STATION_PARAMETERS]
StationName= IO_STATION
[DATA]
0500,10.02.2025 12:00:00
0502,1,100
""")

    # SMET file for the .merge_smet() test
    (update_dir / "merge.smet").write_text("""
[HEADER]
station_name = IO_STATION
fields = timestamp TA
[DATA]
2025-02-09T12:00:00 270.0
""")
    return tmp_path

# --- Test Cases ---

def test_init_with_xr_dataset(sample_xr_dataset):
    """Tests successful initialization with an xarray.Dataset."""
    ds = xsnowDataset(sample_xr_dataset)
    assert ds.data is sample_xr_dataset

def test_getattr_property_forwarding(sample_xsnow_dataset):
    """Tests that property access (e.g., .sizes) is forwarded correctly."""
    assert sample_xsnow_dataset.sizes['layer'] == 3
    assert 'latitude' in sample_xsnow_dataset.coords

def test_getattr_method_forwarding_and_rewrapping(sample_xsnow_dataset):
    """Tests that method calls are forwarded and results are re-wrapped."""
    subset = sample_xsnow_dataset.isel(layer=0)
    assert isinstance(subset, xsnowDataset)
    assert 'layer' not in subset.dims
    assert subset.sizes['time'] == 3

def test_getitem_forwarding(sample_xsnow_dataset):
    """Tests that dictionary-style item access is forwarded correctly."""
    density_da = sample_xsnow_dataset['density']
    assert isinstance(density_da, xr.DataArray)
    assert np.array_equal(density_da.values, [[[100, 200, 300]]])

def test_setitem_forwarding(sample_xsnow_dataset):
    """Tests that dictionary-style item assignment is forwarded correctly."""
    new_var = xr.DataArray([1, 2, 3], dims=['layer'], coords={'layer': sample_xsnow_dataset.layer})
    sample_xsnow_dataset['new_variable'] = new_var
    assert 'new_variable' in sample_xsnow_dataset.data

def test_compute_hs(sample_xsnow_dataset):
    """Tests the compute_hs method for calculating total snow height."""
    ds_with_hs = sample_xsnow_dataset.compute_hs()
    assert 'HS' in ds_with_hs.data
    assert np.isclose(ds_with_hs.data['HS'].item(), 0.3)


def test_compute_z_depth_uses_xsnow_units(sample_xsnow_dataset):
    """Tests z-depth metadata after deriving depth from xsnow-unit heights."""
    ds_with_z = sample_xsnow_dataset.compute_z_depth()

    assert 'z' in ds_with_z.data.coords
    np.testing.assert_allclose(ds_with_z.data['z'].values, [[[-0.2, -0.1, 0.0]]])
    assert ds_with_z.data['z'].attrs['units'] == 'm'
    assert ds_with_z.data['z'].attrs['positive'] == 'up'

def test_compare_processed_hs_against_smet():
    """Compare whether the SMET file variable `HS_mod` agrees with the 
    HS calculated from the cumulative layer `height`

    Note that the two variables often only agree to 1 decimal place. This is an issue of SNOWPACK"""
    ds = single_profile_timeseries()
    # This line causes a KeyError because 16.0 is not in the integer index.
    # sub = ds.sel(realization=0, slope=16.0)

    # First, find the integer index for the desired slope inclination.
    # The .where() method filters the dataset, and .item() extracts the integer index.
    slope_idx_to_select = ds.where(ds.inclination == 16.0, drop=True).slope.item()
    
    # Now, select using the correct integer index.
    sub = ds.sel(realization=0, slope=slope_idx_to_select)
    
    assert np.isclose(sub.HS.values, sub.HS_mod.values, atol=1e-2).all()

def test_dir(sample_xsnow_dataset):
    """Tests __dir__()"""
    ls = dir(sample_xsnow_dataset)
    assert 'to_xarray' in ls

def test_repr(sample_xsnow_dataset):
    """Tests __repr__()"""
    st = sample_xsnow_dataset.__repr__(force_details=True)
    assert "Profiles" in st
    assert "1 total | 1 valid | 1 with HS>0" in st

def test_repr_skips_lazy_details(monkeypatch, sample_xsnow_dataset):
    """__repr__ should avoid computing when details are flagged as lazy."""
    monkeypatch.setattr(xsnowDataset, "_should_compute_repr_detail", lambda self, arr, force_details=False: False)
    st = repr(sample_xsnow_dataset)
    assert "<lazy/uncomputed> valid" in st
    assert "<lazy/uncomputed> with HS>0" in st
    assert ".print(force_details=True)" in st

def test_repr_respects_auto_compute_threshold(monkeypatch, sample_xsnow_dataset):
    """repr should honor the auto-compute threshold and return quickly."""
    max_secs = 0.05
    monkeypatch.setattr(xsnowDataset, "_REPR_MAX_AUTO_COMPUTE_SECONDS", max_secs)
    monkeypatch.setattr(xsnowDataset, "_is_lazy_dataarray", lambda self, arr: True)
    monkeypatch.setattr(xsnowDataset, "_estimate_repr_seconds", lambda self, arr: max_secs * 10)
    monkeypatch.setattr(xsnowDataset, "_compute_int_sum", lambda self, arr: (_ for _ in ()).throw(AssertionError("should not compute")))

    t0 = time.time()
    st = repr(sample_xsnow_dataset)
    elapsed = time.time() - t0

    assert "<lazy/uncomputed> valid" in st
    assert "<lazy/uncomputed> with HS>0" in st
    assert ".print(force_details=True)" in st
    assert elapsed < 2 * max_secs

def test_contains(sample_xsnow_dataset):
    """Tests __contains__()"""
    assert 'location' in sample_xsnow_dataset

def test_to_xarray_returns_xarray_dataset(sample_xsnow_dataset):
    """
    Tests that the to_xarray() method returns a raw xarray.Dataset.
    """
    unwrapped_ds = sample_xsnow_dataset.to_xarray()
    assert isinstance(unwrapped_ds, xr.Dataset)
    assert not isinstance(unwrapped_ds, xsnowDataset)
    assert unwrapped_ds is sample_xsnow_dataset.data

def test_getattr_property_forwarding(sample_xsnow_dataset):
    assert sample_xsnow_dataset.sizes['layer'] == 3
    assert 'latitude' in sample_xsnow_dataset.coords

def test_getattr_method_forwarding_and_rewrapping(sample_xsnow_dataset):
    subset = sample_xsnow_dataset.isel(layer=0)
    assert isinstance(subset, xsnowDataset)
    assert 'layer' not in subset.dims

def test_getitem_forwarding(sample_xsnow_dataset):
    density_da = sample_xsnow_dataset['density']
    assert isinstance(density_da, xr.DataArray)
    assert np.array_equal(density_da.values, [[[100, 200, 300]]])

def test_setitem_forwarding(sample_xsnow_dataset):
    new_var = xr.DataArray([1, 2, 3], dims=['layer'], coords={'layer': sample_xsnow_dataset.layer})
    sample_xsnow_dataset['new_variable'] = new_var
    assert 'new_variable' in sample_xsnow_dataset.data

def test_compute_hs(sample_xsnow_dataset):
    ds_with_hs = sample_xsnow_dataset.compute_hs()
    assert 'HS' in ds_with_hs.data
    assert np.isclose(ds_with_hs.data['HS'].item(), 0.3)
    assert ds_with_hs.data['HS'].attrs['units'] == 'm'
