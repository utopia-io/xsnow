# test_dataset.py
"""
Unit tests for the dataset.py module, focusing on the xsnowDataset class.

This module uses the pytest framework to test the functionality of the
xsnowDataset class, ensuring it initializes correctly, wraps xarray.Dataset
objects, and forwards methods as expected.

---
Key Features of This Test Module
---

This test suite is designed to be fast, reliable, and focused. It achieves
this through two primary techniques:

1.  **Isolation**: The tests for `xsnowDataset` are intentionally isolated
    from the file I/O module (`xsnow_io.py`). This ensures that we are only
    testing the logic within the `xsnowDataset` class itself, not re-testing
    the file-reading code.

2.  **Mocking**: It uses pytest's `monkeypatch` fixture to create a "mock" of
    the `read` function. Instead of performing slow and unpredictable file
    I/O, the tests substitute the `read` function with a dummy one that
    instantly returns a predictable, pre-made `xarray.Dataset`. This makes
    the tests extremely fast and reliable.

---
Setup and Execution
---
These tests assume the same project structure and environment setup as
described in `test_io.py`.

1.  **Install Dependencies**: Ensure the package is installed in editable
    mode with development dependencies:
    `uv pip install -e ".[dev]"`

2.  **Run Tests**: Execute from the project's root directory:
    `pytest`
"""

import pytest
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

# Assuming the project is structured such that xsnow can be imported
from xsnow.dataset import xsnowDataset

# --- Fixtures ---

@pytest.fixture
def mock_xarray_dataset():
    """
    Creates a standard, predictable xarray.Dataset for testing.
    This avoids the need to read actual files, making tests faster and
    more isolated.
    """
    return xr.Dataset(
        data_vars={
            'density': (('location', 'time', 'layer'), np.random.rand(2, 3, 5))
        },
        coords={
            'location': ['station_A', 'station_B'],
            'time': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'layer': np.arange(5)
        }
    )

@pytest.fixture
def mock_read(monkeypatch, mock_xarray_dataset):
    """
    Mocks the `xsnow.xsnow_io.read` function to prevent actual file I/O.
    It makes the read function return a predefined xarray.Dataset instead.
    """
    def mock_return(*args, **kwargs):
        return mock_xarray_dataset
    
    # Replace the real `read` function with our mock version
    monkeypatch.setattr('xsnow.dataset.read', mock_return)


# --- Tests for xsnowDataset ---

def test_init_with_filepath(mock_read):
    """
    Tests that xsnowDataset initializes correctly when given a file path,
    by ensuring it calls the (mocked) read function.
    """
    # The actual path doesn't matter since `read` is mocked
    ds = xsnowDataset("fake/path/to/file.pro")
    assert ds.data is not None
    assert isinstance(ds.data, xr.Dataset)
    assert 'density' in ds.data.data_vars

def test_init_with_list_of_files(mock_read):
    """
    Tests that xsnowDataset initializes correctly with a list of file paths.
    """
    ds = xsnowDataset(["fake/path/1.pro", "fake/path/2.pro"])
    assert ds.data is not None
    assert isinstance(ds.data, xr.Dataset)

def test_init_with_existing_dataset(mock_xarray_dataset):
    """
    Tests that xsnowDataset initializes correctly when passed an existing
    xarray.Dataset object.
    """
    ds = xsnowDataset(mock_xarray_dataset)
    assert ds.data is not None
    # Use assert_identical for a robust comparison
    xr.testing.assert_identical(ds.data, mock_xarray_dataset)

def test_repr(mock_xarray_dataset):
    """
    Tests the __repr__ method for a user-friendly string representation.
    """
    ds = xsnowDataset(mock_xarray_dataset)
    representation = repr(ds)
    assert "xsnowDataset" in representation
    assert "Locations: 2" in representation
    assert "Profiles: 3" in representation
    assert "Time Range: 2025-01-01 to 2025-01-03" in representation

def test_getattr_forwarding_and_rewrapping(mock_xarray_dataset):
    """
    Tests that method calls are forwarded to the underlying xarray.Dataset
    and that the result is re-wrapped into an xsnowDataset instance.
    """
    ds = xsnowDataset(mock_xarray_dataset)

    # Test a method call that returns a new Dataset (e.g., .sel())
    subset = ds.sel(location='station_A')
    
    # Check that the result is an instance of our wrapper class
    assert isinstance(subset, xsnowDataset)
    
    # FIX: When selecting a single item, the dimension is dropped.
    # We verify the coordinate value instead of the dimension size.
    assert 'location' not in subset.sizes
    assert subset.coords['location'].item() == 'station_A'

    # Test accessing a property (e.g., .sizes)
    assert subset.sizes['time'] == 3

def test_to_netcdf(mock_xarray_dataset, tmp_path):
    """
    Tests the to_netcdf method to ensure it saves a file correctly.
    """
    # FIX: Gracefully skip this test if netcdf4 is not installed.
    pytest.importorskip("netCDF4")
    
    ds = xsnowDataset(mock_xarray_dataset)
    output_path = tmp_path / "test_output.nc"
    
    ds.to_netcdf(str(output_path))
    
    # Verify the file was created
    assert output_path.exists()
    
    # Optional: Verify the content is correct by reading it back
    reloaded_ds = xr.open_dataset(output_path)
    xr.testing.assert_identical(mock_xarray_dataset, reloaded_ds)

