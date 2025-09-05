# tests/test_io.py
"""
Unit tests for the xsnow_io module and registered I/O methods.
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

from xsnow import read, xsnowDataset
from xsnow.selectors import TimeSelector
from xsnow.sample_data import snp_gridded_dir, snp_snowobs_dir

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
0501,1,0.1
0502,1,100
"""
    station_b_content = """
[STATION_PARAMETERS]
StationName= station_B
[DATA]
0500,01.01.2025 12:00:00
0501,1,0.2
0502,1,200
"""
    (tmp_path / "station_A.pro").write_text(station_a_content)
    (tmp_path / "station_B.pro").write_text(station_b_content)
    return tmp_path

@pytest.fixture
def data_dir_for_io_methods(tmp_path):
    """Creates a temporary directory with files for testing I/O methods."""
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    update_dir = tmp_path / "update"
    update_dir.mkdir()

    # FIX: Add the height variable (0501) to the .pro files to prevent errors
    # in the downstream compute_hs() and compute_z_depth() methods.
    (initial_dir / "initial.pro").write_text("""
[STATION_PARAMETERS]
StationName= IO_STATION
[DATA]
0500,09.02.2025 12:00:00
0501,1,0.5
0502,1,90
""")

    (update_dir / "update.pro").write_text("""
[STATION_PARAMETERS]
StationName= IO_STATION
[DATA]
0500,10.02.2025 12:00:00
0501,1,0.6
0502,1,100
""")

    (tmp_path / "merge.smet").write_text("""
[HEADER]
station_name = IO_STATION
fields = timestamp TA
[DATA]
2025-02-09T12:00:00 270.0
""")
    return initial_dir, update_dir, tmp_path / "merge.smet"
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

@pytest.fixture
def snp_gridded_sample_dir():
    """Wraps xsnow.sample_data.snp_gridded_ds() into a pytest fixture"""
    return f"{snp_gridded_dir()}/pros/gridded"

# --- Test Cases ---

def test_read_single_pro_file(data_directory):
    """
    Tests reading a single .pro file and verifies the data structure.
    """
    pro_file = data_directory / "test.pro"
    ds = read(pro_file)
    assert isinstance(ds, xsnowDataset)
    assert 'layer' in ds.dims
    assert ds.sizes['time'] == 2
    profile1 = ds.sel(time='2025-01-01 12:00:00')
    expected_density = 350
    actual_density = profile1['density'].isel(layer=2).item()
    assert np.isclose(actual_density, expected_density)

def test_read_from_directory(data_directory):
    """
    Tests reading all files from a directory and merging the data.
    """
    ds = read(data_directory)
    assert isinstance(ds, xsnowDataset)
    assert 'density' in ds.data.data_vars
    assert 'TA' in ds.data.data_vars
    assert 'layer' not in ds.data['TA'].dims

def test_read_with_time_selector_at_source(data_directory):
    """
    Tests at-the-source filtering using a TimeSelector.
    """
    time_filter = TimeSelector(sel=slice('2025-01-02 00:00:00', '2025-01-02 23:59:59'))
    ds = read(data_directory, time=time_filter)
    assert isinstance(ds, xsnowDataset)
    assert ds.sizes['time'] == 1
    loaded_time = pd.to_datetime(ds.time.values[0])
    assert loaded_time.day == 2

def test_read_with_max_layers_kwarg(data_directory):
    """
    Tests passing parser-specific keyword arguments like `max_layers`.
    """
    ds = read(data_directory, max_layers=3)
    assert isinstance(ds, xsnowDataset)
    assert ds.sizes['layer'] == 3
    last_layer_density = ds.data['density'].sel(time='2025-01-02 12:00:00').isel(layer=-1).item()
    assert np.isclose(last_layer_density, 340)

def test_read_empty_directory(tmp_path):
    """
    Tests that reading from an empty directory returns None without error.
    """
    ds = read(tmp_path)
    assert ds is None

def test_read_with_integer_location_index(multi_station_dir):
    """
    Tests that read() creates an integer index for the location dimension.
    """
    ds = read(multi_station_dir)
    assert np.issubdtype(ds.location.dtype, np.integer)
    np.testing.assert_array_equal(ds.location.values, [0, 1])
    expected_mapping = {0: 'station_A', 1: 'station_B'}
    assert 'mapping' in ds.location.attrs
    assert ds.location.attrs['mapping'] == expected_mapping

def test_read_sample_data_snp_gridded(snp_gridded_sample_dir):
    ds = read(snp_gridded_sample_dir)
    # TODO: write some assertions once the reading of the ds is successful

def test_read_sample_data_snp_snowobs():
    snowobs_dir = snp_snowobs_dir()
    ds = read([
        f"{snowobs_dir}/pros/snowobs/",
        f"{snowobs_dir}/pits/",
    ])
    assert isinstance(ds, xsnowDataset)
    # TODO: write some more assertions once the reading of the ds is successful,
    #       for example, checking realizations

def test_to_pro_max_files_raises_error(io_test_dataset):
    """
    Tests that .to_pro() raises a ValueError if the number of profiles
    exceeds the max_files limit.
    """
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

def test_update_method(data_dir_for_io_methods):
    """
    Tests the .update() method to ensure it loads and appends new data.
    """
    initial_dir, update_dir, _ = data_dir_for_io_methods
    
    ds = read(initial_dir)
    assert ds.sizes['time'] == 1
    
    ds_updated = ds.update(update_dir)
    
    assert isinstance(ds_updated, xsnowDataset)
    assert ds_updated.sizes['time'] == 2
    
    new_profile = ds_updated.sel(time='2025-02-10 12:00:00')
    assert np.isclose(new_profile.density.item(), 100)
    
def test_merge_smet_method(data_dir_for_io_methods):
    """
    Tests the .merge_smet() method to ensure it merges SMET data correctly.
    """
    initial_dir, _, smet_file = data_dir_for_io_methods
    ds = read(initial_dir)
    assert 'TA' not in ds.data.data_vars
    ds_merged = ds.merge_smet(smet_file)
    assert isinstance(ds_merged, xsnowDataset)
    assert 'density' in ds_merged.data.data_vars
    assert 'TA' in ds_merged.data.data_vars
    profile = ds_merged.sel(time='2025-02-09 12:00:00')
    assert np.isclose(profile.TA.item(), 270.0)