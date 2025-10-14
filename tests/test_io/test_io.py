# tests/test_io.py
"""
Unit tests for the xsnow_io module and registered I/O methods.
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import re
import warnings

from xsnow import read, xsnowDataset
from xsnow.selectors import TimeSelector, SlopeSelector, RealizationSelector
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

@pytest.fixture
def gridded_data_integrity_check(snp_gridded_sample_dir):
    """
    Checks for consistent lat/lon values for each station name in a directory.
    If an inconsistency is found, the test is skipped with a warning.
    """
    directory_path = Path(snp_gridded_sample_dir)
    station_coords = {}
    pro_files = list(directory_path.glob("*.pro"))

    for pro_file in pro_files:
        with open(pro_file, 'r', errors='ignore') as f:
            content = f.read()
            station_match = re.search(r"StationName=\s*(.*)", content)
            lat_match = re.search(r"Latitude=\s*([0-9.-]+)", content)
            lon_match = re.search(r"Longitude=\s*([0-9.-]+)", content)

            if not (station_match and lat_match and lon_match):
                continue

            station_name = station_match.group(1).strip()
            latitude = float(lat_match.group(1))
            longitude = float(lon_match.group(1))

            if station_name in station_coords:
                stored_lat, stored_lon, first_file = station_coords[station_name]
                if not np.isclose(latitude, stored_lat) or not np.isclose(longitude, stored_lon):
                    msg = (
                        f"Inconsistent coordinates found for station '{station_name}'.\n"
                        f"  - File '{first_file.name}' has (Lat, Lon): ({stored_lat}, {stored_lon})\n"
                        f"  - File '{pro_file.name}' has (Lat, Lon): ({latitude}, {longitude})"
                    )
                    warnings.warn(msg)
            else:
                station_coords[station_name] = (latitude, longitude, pro_file)

    return directory_path

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


def test_read_sample_data_snp_gridded(gridded_data_integrity_check):
    """
    This test reads a gridded dataset after verifying its integrity.
    It confirms that all expected stations are loaded correctly.
    If the data is inconsistent, the gridded_data_integrity_check fixture
    will skip this test.
    """
    verified_dir = gridded_data_integrity_check
    ds = read(verified_dir)

    # 1. Basic dataset validation
    assert ds is not None
    assert isinstance(ds, xsnowDataset)
    assert 'location' in ds.dims

    # 2. Verify all expected stations are present
    location_map = ds.location.attrs.get('mapping', {})
    loaded_stations = set(location_map.values())
    expected_stations = {"VIR1A", "VIR2A", "VIR3A", "VIR4A", "VIR5A"}

    assert ds.sizes['location'] == len(expected_stations)
    assert expected_stations.issubset(loaded_stations)

    # 3. Check a specific station for correct dimension merging (e.g., slope)
    station_names = list(location_map.values())
    vir1a_ds = ds.sel(location=station_names.index('VIR1A'))
    assert 'slope' in vir1a_ds.dims
    assert vir1a_ds.sizes['slope'] > 1
    

def test_read_sample_data_snp_snowobs():
    
    snowobs_path = Path(snp_snowobs_dir())
    
    pro_dir = snowobs_path / "pros" / "snowobs"
    pit_dir = snowobs_path / "pits"

    ds = read([pro_dir, pit_dir])

    assert (isinstance(ds, xsnowDataset))
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
    assert 'TA' in ds_merged.data.data_vars
    profile = ds_merged.sel(time='2025-02-09 12:00:00')
    assert np.isclose(profile.TA.item(), 270.0)

def test_read_merge_smet_snp_snowobs():
    """
    Tests merging SMET data with the snow observation sample dataset.
    Skips if the sample data is not available.
    """
    snowobs_dir = Path(snp_snowobs_dir())
    
    pro_dir = snowobs_dir / "pros" / "snowobs"
    pit_dir = snowobs_dir / "pits"

    ds = read([
        pro_dir,
        pit_dir,
    ])

    assert isinstance(ds, xsnowDataset)

    # The rest of the test proceeds only if ds is a valid dataset
    assert isinstance(ds, xsnowDataset)
    ds_merged = ds.merge_smet([
        snowobs_dir / "smets" / "snowobs" / "nowcast",
        snowobs_dir / "smets" / "snowobs" /"forecast"
    ], realization=1, var_prefix='input.')
    
    # Add assertions to verify the merge was successful
    assert isinstance(ds_merged, xsnowDataset)
    assert ('input.TA' in ds_merged.data)
    
@pytest.fixture(scope="module")
def pro_data_with_soil():
    """Provides .pro file content that includes soil layers."""
    return """
[STATION_PARAMETERS]
StationName= SOIL_STATION
[DATA]
0500,01.01.2025 12:00:00
0501,4,0.1,0.2,0.3,0.4
0502,4,150,250,350,900
0513,4,311,421,81,82
"""

@pytest.fixture
def multi_slope_directory(tmp_path):
    """
    Creates a temporary directory with two .pro files for the same station
    but with different slope angles (15 and 30 degrees).
    """
    station_dir = tmp_path / "station_A"
    station_dir.mkdir()

    pro_content_15 = """
[STATION_PARAMETERS]
StationName= STATION_A
SlopeAngle= 15.0
[DATA]
0500,01.01.2025 12:00:00
0501,1,0.5
0502,1,100
"""
    pro_content_30 = """
[STATION_PARAMETERS]
StationName= STATION_A
SlopeAngle= 30.0
[DATA]
0500,01.01.2025 12:00:00
0501,1,0.8
0502,1,150
"""
    (station_dir / "station_A_slope15.pro").write_text(pro_content_15)
    (station_dir / "station_A_slope30.pro").write_text(pro_content_30)
    return station_dir


def test_read_pro_with_soil_removal(tmp_path, pro_data_with_soil):
    """Tests that soil layers are correctly removed by default."""
    pro_file = tmp_path / "soil_test.pro"
    pro_file.write_text(pro_data_with_soil)
    
    # Read with default behavior (remove_soil=True)
    ds = read(pro_file)
    assert 'layer' in ds.dims
    assert ds.sizes['layer'] == 2  # Soil layers should be gone
    assert 'grain_type' in ds.data
    # Check that remaining grain types are not soil
    assert not np.any((ds.data['grain_type'].values == 81) | (ds.data['grain_type'].values == 82))

    # Read with soil removal disabled
    ds_with_soil = read(pro_file, remove_soil=False)
    assert ds_with_soil.sizes['layer'] == 4 # All layers should be present
    
def test_read_with_slope_selector(multi_slope_directory):
    """
    Tests that passing a SlopeSelector to read() correctly filters the
    dataset by the slope's integer index.
    """
    # The `read` function should create a dataset with two slopes, indexed 0 and 1.
    # We expect the slopes to be sorted, so 15.0 is at index 0 and 30.0 is at index 1.
    
    # Test selecting a single slope by its integer index
    slope_selector_single = SlopeSelector(sel=0)
    ds_single = read(multi_slope_directory, slope=slope_selector_single)

    assert ds_single is not None
    assert 'slope' not in ds_single.dims
    assert 'slope' in ds_single.coords
    assert int(ds_single.slope) == 0
    # Verify we got the correct slope by checking the associated inclination coordinate
    assert np.isclose(ds_single.slope_inclination.item(), 15.0)

    # Test selecting a slice of slopes
    slope_selector_slice = SlopeSelector(sel=slice(1)) # Selects index 0
    ds_slice = read(multi_slope_directory, slope=slope_selector_slice)
    
    assert ds_slice is not None
    assert ds_slice.sizes['slope'] == 1
    assert np.isclose(ds_slice.slope_inclination.item(), 15.0)
    
    # Test selecting an out-of-bounds index, which should be empty
    slope_selector_empty = SlopeSelector(sel=99)
    ds_empty = read(multi_slope_directory, slope=slope_selector_empty)
    
    # An empty selection on a dimension results in a size of 0 for that dim
    assert ds_empty is not None and ds_empty.sizes['slope'] == 0


def test_read_with_realization_selector(multi_slope_directory):
    """
    Tests that passing a RealizationSelector to read() correctly filters the
    dataset. Since .pro files are read as realization 0, this tests
    selecting for 0 and for a non-existent realization.
    """
    # Test selecting the default realization (0)
    realization_selector_present = RealizationSelector(sel=0)
    ds_present = read(multi_slope_directory, realization=realization_selector_present)

    assert ds_present is not None
    assert 'realization' not in ds_present.dims
    assert 'realization' in ds_present.coords
    assert int(ds_present.realization) == 0
    assert ds_present.realization.item() == 0

    # Test selecting a non-existent realization, which should be empty
    realization_selector_absent = RealizationSelector(sel=99)
    ds_absent = read(multi_slope_directory, realization=realization_selector_absent)

    # An empty selection on a dimension results in a size of 0 for that dim
    assert ds_absent is not None and ds_absent.sizes['realization'] == 0
    
def test_all_station_files_merge_correctly(tmp_path):
    """
    Tests the comprehensive merging of various station files by the `read` function.

    This test ensures that when the `read` function is pointed to a source
    containing multiple file types (.pro from modeled runs, .pro from manual
    pits, and .smet) for the same stations, it correctly groups and merges them
    into a single dataset per station.

    The test performs the following steps:
    1.  Sets up paths to the 'snowobs' sample data, which contains all three
        file types for two distinct stations.
    2.  Calls `read()` on this temporary directory.
    3.  Verifies the merge by first asserting that the total number of
        locations in the final dataset is 2, not the total number of
        individual files. This confirms the grouping logic worked.
    4.  It then inspects each of the two merged locations to confirm that they
        contain variables originating from both the profile files (`density`)
        and the time-series files (`TA`), proving the content was merged.
    """
    # 1. Setup paths to all original sample data directories
    snowobs_path = Path(snp_snowobs_dir())
    modeled_dir = snowobs_path / "pros" / "snowobs"
    pit_dir = snowobs_path / "pits"

    # 2. Read from the temporary directory using the original 'source' argument
    ds = read([modeled_dir, pit_dir])

    # 3. Assertions
    assert ds is not None
    assert isinstance(ds, xsnowDataset)

    # Assert that all files for the two stations were merged into two locations
    assert ds.sizes['location'] == 2

    location_map = ds.location.attrs['mapping']
    name_map = {v: k for k, v in location_map.items()}

    # Confirm that both dataset contain variables originating from both the profile files
    for station_name, station_index in name_map.items():
        
        station_ds = ds.sel(location=station_index)

        assert 'density' in station_ds.data, f"Variable 'density' missing for {station_name}"
        assert 'layer' in station_ds.data['density'].dims, f"Dimension 'layer' missing for {station_name}"
        assert 'TA' in station_ds.data, f"Variable 'TA' missing for {station_name}"
        assert 'layer' not in station_ds.data['TA'].dims, f"Dimension 'layer' present in 'TA' for {station_name}"
        

def test_read_real_pro_with_mixed_sh(snp_gridded_sample_dir):
    """
    Tests reading a real .pro file containing profiles with and without surface hoar.
    It verifies that the SH layer is correctly added or skipped based on the data.
    """
    pro_file = Path(snp_gridded_sample_dir) / "VIR3A3.pro"
    ds = read(pro_file)
    assert ds is not None

    # Flags to confirm we found at least one of each profile type
    found_sh_profile = False
    found_no_sh_profile = False

    # Iterate through each timestamp in the dataset
    for timestamp in ds.time.values:
        profile = ds.sel(time=timestamp).squeeze()
        
        # Drop padding to get the true profile for this timestamp
        true_profile = profile.dropna(dim='layer', how='all')
        if true_profile.sizes.get('layer', 0) == 0:
            continue # Skip empty profiles

        top_layer = true_profile.isel(layer=-1)

        # Check if this is a surface hoar layer (identified by NaN temperature)
        if np.isnan(top_layer['temperature'].item()):
            # This is an SH profile, verify its properties
            assert np.isclose(top_layer['grain_type'].item(), 660)
            assert np.isclose(top_layer['density'].item(), 100)
            assert np.isclose(top_layer['grain_size'].item(), 0.5)
            found_sh_profile = True
        
        # FIX: A valid, non-SH profile is one where the temperature is NOT NaN.
        elif not np.isnan(top_layer['temperature'].item()):
            assert not np.isclose(top_layer['grain_type'].item(), 660, equal_nan=True)
            found_no_sh_profile = True

    # After checking all profiles, assert that we found both types
    assert found_sh_profile, "No profile with surface hoar was found in the test file."
    assert found_no_sh_profile, "No valid profile without surface hoar was found."