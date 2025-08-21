# tests/test_io.py
"""
Unit tests for the xsnow_io module.

This test suite verifies the functionality of the primary `xsnow.read()`
function, ensuring it can correctly parse different file formats, handle
various source inputs, and apply at-the-source filtering as expected.
"""

import pytest
import xarray as xr
from pathlib import Path
import pandas as pd
import numpy as np

# Import the functions and classes to be tested
from xsnow import read
from xsnow.dataset import xsnowDataset
from xsnow.selectors import TimeSelector

# --- Test Data Fixtures ---

@pytest.fixture(scope="module")
def pro_data_content():
    """Provides the text content for a standard .pro file."""
    return """
[STATION_PARAMETERS]
StationName= TEST_STATION
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
0500,02.01.2025 12:00:00
0501,4,0.1,0.2,0.3,0.4
0502,4,140,240,340,400
"""

@pytest.fixture(scope="module")
def smet_data_content():
    """Provides the text content for a standard .smet file."""
    return """
SMET 1.1 ASCII
[HEADER]
station_name = TEST_STATION
latitude = 46.5
longitude = 10.5
altitude = 2500
fields = timestamp TA HS
[DATA]
2025-01-01T12:00:00 270.15 0.3
2025-01-02T12:00:00 271.15 0.4
"""

@pytest.fixture
def data_directory(tmp_path, pro_data_content, smet_data_content):
    """Creates a temporary directory with a .pro and a .smet file."""
    pro_file = tmp_path / "test.pro"
    pro_file.write_text(pro_data_content)
    
    smet_file = tmp_path / "test.smet"
    smet_file.write_text(smet_data_content)
    
    return tmp_path

@pytest.fixture
def multi_station_dir(tmp_path):
    """Creates a directory with .pro files for two different stations."""
    station_a_content = """
[STATION_PARAMETERS]
StationName= station_A
[DATA]
0500,01.01.2025 12:00:00
0502,1,100
"""
    station_b_content = """
[STATION_PARAMETERS]
StationName= station_B
[DATA]
0500,01.01.2025 12:00:00
0502,1,200
"""
    (tmp_path / "station_A.pro").write_text(station_a_content)
    (tmp_path / "station_B.pro").write_text(station_b_content)
    return tmp_path

@pytest.fixture
def io_test_dataset():
    """Creates a minimal xsnowDataset for I/O tests like to_pro."""
    data = {
        'height': (('location', 'time', 'layer'), np.array([[[0.1, 0.2, 0.3]]])),
        'density': (('location', 'time', 'layer'), np.array([[[100, 200, 300]]]))
    }
    coords = {
        'location': [0],
        'time': [pd.to_datetime('2025-01-15')],
        'layer': [0, 1, 2]
    }
    return xsnowDataset(xr.Dataset(data, coords))

# --- Test Cases ---

def test_read_with_integer_location_index(multi_station_dir):
    """
    Tests that read() creates an integer index for the location dimension.
    """
    ds = read(multi_station_dir)
    
    # 1. Check that the coordinate dtype is integer
    assert np.issubdtype(ds.location.dtype, np.integer)
    
    # 2. Check that the values are the expected integer indices
    np.testing.assert_array_equal(ds.location.values, [0, 1])
    
    # 3. Check that the mapping from integer to name is stored in attributes
    expected_mapping = {0: 'station_A', 1: 'station_B'}
    assert 'mapping' in ds.location.attrs
    assert ds.location.attrs['mapping'] == expected_mapping

def test_to_pro_max_files_raises_error(io_test_dataset): # Uses the new fixture
    """
    Tests that .to_pro() raises a ValueError if the number of profiles
    exceeds the max_files limit.
    """
    # The sample dataset has 1 timestamp (profile)
    with pytest.raises(ValueError, match="exceeds the specified maximum"):
        io_test_dataset.to_pro("test.pro", max_files=0)

def test_to_smet_max_files_raises_error(multi_station_dir):
    """
    Tests that .to_smet() raises a ValueError if the number of locations
    exceeds the max_files limit.
    """
    ds = read(multi_station_dir)
    with pytest.raises(ValueError, match="exceeds the specified maximum"):
        ds.to_smet("test.smet", max_files=1)
        
def test_read_single_pro_file(data_directory):
    """
    Tests reading a single .pro file and verifies the data structure.
    """
    pro_file = data_directory / "test.pro"
    ds = read(pro_file)

    assert isinstance(ds, xsnowDataset)
    
    # Check for 'layer' dimension
    assert 'layer' in ds.dims
    assert ds.sizes['time'] == 2
    
    # Check a data point at the surface (top-most layer)
    profile1 = ds.sel(time='2025-01-01 12:00:00')
    expected_density = 350
    # The first profile has 3 layers (0, 1, 2), so the last is at index 2
    actual_density = profile1['density'].isel(layer=2).item()
    assert np.isclose(actual_density, expected_density)

def test_read_from_directory(data_directory):
    """
    Tests reading all files from a directory and merging the data.
    """
    ds = read(data_directory)
    assert isinstance(ds, xsnowDataset)
    
    # Check that variables from both .pro and .smet files are present
    assert 'density' in ds.data.data_vars  # From .pro
    assert 'TA' in ds.data.data_vars       # From .smet
    
    # Check that the surface flux (TA) does not have a layer dimension
    assert 'layer' not in ds.data['TA'].dims

def test_read_with_time_selector_at_source(data_directory):
    """
    Tests at-the-source filtering using a TimeSelector.
    
    This test ensures that the parser only reads the data within the specified
    time slice, which is more memory-efficient.
    """
    # Make the time slice cover the entire day to include the 12:00 profile.
    time_filter = TimeSelector(sel=slice('2025-01-02 00:00:00', '2025-01-02 23:59:59'))
    
    ds = read(data_directory, time=time_filter)

    assert isinstance(ds, xsnowDataset)
    
    # Verify that only one timestamp was loaded
    assert ds.sizes['time'] == 1
    
    # Verify it's the correct timestamp
    loaded_time = pd.to_datetime(ds.time.values[0])
    assert loaded_time.day == 2

def test_read_with_max_layers_kwarg(data_directory):
    """
    Tests passing parser-specific keyword arguments like `max_layers`.
    """
    # The test data has a max of 4 layers. We'll cap it at 3.
    ds = read(data_directory, max_layers=3)

    assert isinstance(ds, xsnowDataset)
    
    # The 'layer' dimension should be capped at 3
    assert ds.sizes['layer'] == 3
    
    last_layer_density = ds.data['density'].sel(time='2025-01-02 12:00:00').isel(layer=-1).item()
    assert np.isclose(last_layer_density, 340)

def test_read_empty_directory(tmp_path):
    """
    Tests that reading from an empty directory returns None without error.
    """
    ds = read(tmp_path)
    assert ds is None