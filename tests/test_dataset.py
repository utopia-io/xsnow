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
from xsnow.xsnow_io import read_smet

# --- Fixtures ---

@pytest.fixture
def mock_xarray_dataset():
    """
    Creates a standard, predictable xarray.Dataset for testing.
    This avoids the need to read actual files, making tests faster and
    more isolated.
    """
    # Ensure all data variables have consistent dimensions.
    # The height array is now broadcast to the full (2, 3, 10) shape.
    height_array = np.broadcast_to(np.linspace(0, 1, 10), (2, 3, 10))
    
    return xr.Dataset(
        data_vars={
            'density': (('location', 'time', 'layer'), np.full((2, 3, 10), 250.0)),
            'height': (('location', 'time', 'layer'), height_array),
            'HS': (('location', 'time'), np.random.rand(2, 3)) # Add a timeseries variable
        },
        coords={
            'location': ['station_A', 'station_B'],
            'time': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'layer': np.arange(10),
            'latitude': ('location', [45.0, 46.0]),
            'longitude': ('location', [10.0, 11.0]),
            'altitude': ('location', [2000, 2100]),
            'slope': ('location', [25.0, 30.0]),
            'slope_aspect': ('location', [90.0, 180.0])
        }
    )

@pytest.fixture
def mock_smet_file(tmp_path):
    """Creates a dummy SMET file for testing merging."""
    content = """
SMET 1.1 ASCII
[HEADER]
station_name     = station_A
latitude         = 45.0
longitude        = 10.0
altitude         = 2000
nodata           = -999
fields           = timestamp VW TA
[DATA]
2025-01-01T00:00:00 5.5 270.0
2025-01-02T00:00:00 6.0 271.0
"""
    file_path = tmp_path / "station_A.smet"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def mock_read(monkeypatch, mock_xarray_dataset):
    """
    Mocks the `xsnow.xsnow_io.read` function to prevent actual file I/O.
    It makes the read function return a predefined xarray.Dataset instead.
    """
    def mock_return(source, *args, **kwargs):
        # If the source is a smet file, read it for real
        if isinstance(source, str) and '.smet' in source:
            return read_smet(source)
        # Otherwise, return the mock profile data
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
    
    # When selecting a single item, the dimension is dropped.
    # We verify the coordinate value instead of the dimension size.
    assert 'location' not in subset.sizes
    assert subset.coords['location'].item() == 'station_A'

    # Test accessing a property (e.g., .sizes)
    assert subset.sizes['time'] == 3

def test_merge_smet(mock_xarray_dataset, mock_smet_file):
    """Tests merging of SMET data into an existing dataset."""
    ds = xsnowDataset(mock_xarray_dataset)
    merged_ds = ds.merge_smet(str(mock_smet_file))
    
    assert isinstance(merged_ds, xsnowDataset)
    # Check for a variable from the original dataset
    assert 'density' in merged_ds.data
    # Check for variables from the new SMET file
    assert 'VW' in merged_ds.data
    assert 'TA' in merged_ds.data
    # Check if a value was merged correctly
    assert merged_ds.data['VW'].sel(location='station_A', time='2025-01-01').item() == 5.5

def test_to_xarray(mock_xarray_dataset):
    """
    Tests the to_xarray method
    """
    ds = xsnowDataset(mock_xarray_dataset)
    xrds = ds.to_xarray

    assert isinstance(xrds, xr.core.dataset.Dataset)

def test_to_netcdf(mock_xarray_dataset, tmp_path):
    """
    Tests the to_netcdf method to ensure it saves a file correctly.
    """
    # Gracefully skip this test if netcdf4 is not installed.
    pytest.importorskip("netCDF4")
    
    ds = xsnowDataset(mock_xarray_dataset)
    output_path = tmp_path / "test_output.nc"
    
    ds.to_netcdf(str(output_path))
    
    # Verify the file was created
    assert output_path.exists()
    
    # Optional: Verify the content is correct by reading it back
    reloaded_ds = xr.open_dataset(output_path)
    xr.testing.assert_identical(mock_xarray_dataset, reloaded_ds)

def test_to_pro(mock_xarray_dataset, tmp_path):
    """Tests writing to .pro files."""
    ds = xsnowDataset(mock_xarray_dataset)
    output_dir = tmp_path / "pro_output"
    
    # Total profiles = 2 locations * 3 timestamps = 6
    ds.to_pro(str(output_dir))
    
    created_files = list(output_dir.glob("*.pro"))
    assert len(created_files) == 6
    
    # Test the max_profiles guard
    with pytest.raises(ValueError):
        ds.to_pro(str(output_dir), max_profiles=5)

def test_to_smet(mock_xarray_dataset, tmp_path):
    """Tests writing to .smet files."""
    ds = xsnowDataset(mock_xarray_dataset)
    output_dir = tmp_path / "smet_output"
    
    ds.to_smet(str(output_dir))
    
    created_files = list(output_dir.glob("*.smet"))
    assert len(created_files) == 2
    
    # Test the max_files guard
    ds_large = xsnowDataset(mock_xarray_dataset.isel(location=slice(0,1))) # create a single location dataset
    with pytest.raises(ValueError):
        ds.to_smet(str(output_dir), max_files=1)

# --- Use Case Examples for Extensions ---

# --- Example 1: Simple Extension (Standalone Function) ---

def add_slab_density(ds: xsnowDataset, slab_thickness: float = 0.3) -> xsnowDataset:
    """
    Calculates the average density of the top slab of the snowpack.
    This is an example of a simple extension that adds a new variable.
    """
    total_height = ds.data['height'].max(dim='layer')
    slab_layers = ds.data['density'].where(ds.data['height'] >= total_height - slab_thickness)
    slab_density = slab_layers.mean(dim='layer')
    
    new_xarray_ds = ds.data.assign(slab_density=slab_density)
    
    # The key step: re-wrap the modified xarray.Dataset in our class
    return xsnowDataset(new_xarray_ds)

def test_simple_extension_function(mock_xarray_dataset):
    """
    Tests the simple extension pattern by applying the add_slab_density function.
    """
    ds = xsnowDataset(mock_xarray_dataset)
    
    # Apply the external function
    extended_ds = add_slab_density(ds)
    
    # Verify the result
    assert isinstance(extended_ds, xsnowDataset)
    assert 'slab_density' in extended_ds.data
    assert extended_ds.data['slab_density'].sizes == {'location': 2, 'time': 3}
    assert extended_ds.data['slab_density'].isel(location=0, time=0).item() == 250.0

# --- Example 2: Complex Extension (Decorator Class) ---

class DatasetDecorator:
    """Generic decorator base class to handle attribute forwarding and re-wrapping."""
    def __init__(self, base: xsnowDataset):
        # The decorator should have its own reference to the data,
        # which can be modified by subclasses.
        self.data = base.data

    def __getattr__(self, name):
        # Forward attribute access to the internal xarray.Dataset
        attr = getattr(self.data, name)
        if callable(attr):
            def method_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, xr.Dataset):
                    # Re-wrap the result in a new base class, then decorate it
                    new_base = xsnowDataset(result)
                    return self.__class__(new_base)
                return result
            return method_wrapper
        return attr

class HazardChartData(DatasetDecorator):
    """
    A decorator that adds a new 'hazardChartElement' dimension and data.
    This is an example of a complex extension.
    """
    def __init__(self, base: xsnowDataset):
        super().__init__(base)
        # Only compute the new data if it's not already present.
        # This prevents re-computation and merge errors when re-wrapping a subset.
        if 'hazard_value' not in self.data.data_vars:
            self.data = self._compute_hazard_elements()

    def _compute_hazard_elements(self):
        """
        Computes the hazard elements. This method is now robust to subsetting,
        as it dynamically determines the shape of the new variable.
        """
        num_elements = 5
        
        # Identify dimensions to preserve from the base dataset
        base_dims = [dim for dim in ['location', 'time'] if dim in self.data.dims]
        shape = [self.data.sizes[dim] for dim in base_dims] + [num_elements]
        dims = base_dims + ['hazardChartElement']

        hazard_values = xr.DataArray(
            np.random.rand(*shape),
            dims=dims
        )
        # Add the coordinate for the new dimension
        hazard_values = hazard_values.assign_coords(hazardChartElement=np.arange(num_elements))

        # Merge the new data into the original dataset
        return self.data.merge(hazard_values.rename('hazard_value'))

def test_complex_extension_decorator(mock_xarray_dataset):
    """
    Tests the complex extension pattern by applying the HazardChartData decorator.
    """
    ds = xsnowDataset(mock_xarray_dataset)
    
    # Apply the decorator
    decorated_ds = HazardChartData(ds)
    
    # Verify the result
    assert isinstance(decorated_ds, HazardChartData)
    assert 'hazardChartElement' in decorated_ds.sizes
    assert 'hazard_value' in decorated_ds.data
    assert decorated_ds.sizes['hazardChartElement'] == 5
    
    # Verify that the original functionality is still present
    subset = decorated_ds.sel(location='station_A')
    assert isinstance(subset, HazardChartData) # It remains decorated
    assert 'location' not in subset.sizes
