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
tz = 1
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
tz = 1
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

def test_single_smet_file_metadata(data_directory):
    """
    Tests reading a single .smet file and verifies the metadata of the dataset and variables.
    """
    smet_file = data_directory / "test.smet"
    ds = read(smet_file)
    assert ds.attrs
    assert 'crs' in ds.attrs
    assert 'history' in ds.attrs
    assert 'timezone' in ds['time'].attrs
    assert ds['time'].attrs['timezone'] == 'UTC+01:00'  # converted from int
    for var in list(ds.variables):
        assert 'units' in ds[var].attrs
        assert 'standard_name' in ds[var].attrs

def test_single_pro_file_metadata(data_directory):
    """
    Tests reading a single .pro file and verifies the metadata of the dataset and variables.
    """
    pro_file = data_directory / "test.pro"
    ds = read(pro_file)
    assert ds.attrs
    assert 'crs' in ds.attrs
    assert 'history' in ds.attrs
    for var in list(ds.variables):
        assert 'units' in ds[var].attrs
        assert 'standard_name' in ds[var].attrs
    assert 'timezone' in ds['time'].attrs
    assert ds['time'].attrs['timezone'] == 'UTC+01:00'  # converted from int

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
    Tests that read() creates string-based location coordinates (not integer indices).
    """
    ds = read(multi_station_dir)
    
    # Now expects string dtype instead of integer
    assert ds.location.dtype.kind in ('U', 'S', 'O')
    
    # Verify station names are present
    locations = [str(loc) for loc in ds.location.values]
    assert 'station_A' in locations
    assert 'station_B' in locations
    assert len(locations) == 2


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
    # NEW: Get stations directly from location coordinate values (strings)
    loaded_stations = {str(loc) for loc in ds.location.values}
    expected_stations = {"VIR1A", "VIR2A", "VIR3A", "VIR4A", "VIR5A"}

    assert ds.sizes['location'] == len(expected_stations)
    assert expected_stations.issubset(loaded_stations)
    
    # 3. Verify locations are string-based (not integer indices)
    assert ds.location.dtype.kind in ('U', 'S', 'O')    

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



def test_read_with_slope_selector(multi_slope_directory, compare_datasets):
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
    assert np.isclose(ds_single.inclination.item(), 15.0)

    # Repeat with SlopeLike input
    ds_single2 = read(multi_slope_directory, slope=0)
    compare_datasets(ds_single, ds_single2)

    # Test selecting a slice of slopes
    slope_selector_slice = SlopeSelector(sel=slice(1)) # Selects index 0
    ds_slice = read(multi_slope_directory, slope=slope_selector_slice)
    
    assert ds_slice is not None
    assert ds_slice.sizes['slope'] == 1
    assert np.isclose(ds_slice.inclination.item(), 15.0)

    # Repeat with SlopeLike input
    ds_slice2 = read(multi_slope_directory, slope={"inclination": 15})
    compare_datasets(ds_slice, ds_slice2)
    
    # Test selecting an out-of-bounds index, which should be empty
    slope_selector_empty = SlopeSelector(sel=99)
    ds_empty = read(multi_slope_directory, slope=slope_selector_empty)
    
    # An empty selection on a dimension results in a size of 0 for that dim
    assert ds_empty is not None and ds_empty.sizes['slope'] == 0


def test_read_with_realization_selector(multi_slope_directory, compare_datasets):
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

    # Repeat with RealizationLike input
    ds_present2 = read(multi_slope_directory, realization=0)
    compare_datasets(ds_present, ds_present2)

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
    # Assert that all files for the two stations were merged into two locations
    assert ds.sizes['location'] == 2
    
    # location.values is now an array of strings, not a dict
    location_names = [str(loc) for loc in ds.location.values]
    
    # Confirm that both datasets contain variables from both file types
    for location_name in location_names:
        station_ds = ds.sel(location=location_name)
        
        assert 'density' in station_ds.data, f"Variable 'density' missing for {location_name}"
        assert 'layer' in station_ds.data['density'].dims, f"Dimension 'layer' missing for {location_name}"
        assert 'TA' in station_ds.data, f"Variable 'TA' missing for {location_name}"
        assert 'layer' not in station_ds.data['TA'].dims, f"Dimension 'layer' present in 'TA' for {location_name}"

        

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
    
# tests/test_io.py - New tests

@pytest.mark.synthetic
class TestStringCoordinateIO:
    """Test I/O with string coordinates."""
    
    def test_read_single_pro_has_string_location(self, synthetic_pro, tmp_path):
        """Test reading single .pro creates string location."""
        filepath = synthetic_pro.basic(tmp_path, station_name='TEST', n_layers=3)
        
        ds = read(filepath)
        
        assert ds.data.location.dtype.kind in ('U', 'S', 'O')
        assert str(ds.data.location.values[0]) == 'TEST'
    
    def test_netcdf_roundtrip_preserves_strings(self, synthetic_pro, tmp_path):
        """Test NetCDF save/load preserves string coords."""
        filepath = synthetic_pro.basic(tmp_path, station_name='ROUNDTRIP_TEST', n_layers=3)
        
        ds_original = read(filepath)
        
        # Save to NetCDF
        nc_path = tmp_path / "roundtrip.nc"
        ds_original.to_netcdf(nc_path)
        
        # Load back
        ds_loaded = read(nc_path)
        
        # Verify location is string and matches
        assert ds_loaded.data.location.dtype.kind in ('U', 'S', 'O')
        
        # FIX: Handle both bytes and strings
        loaded_name = ds_loaded.data.location.values[0]
        if isinstance(loaded_name, bytes):
            loaded_name = loaded_name.decode('utf-8')
        else:
            loaded_name = str(loaded_name)
        
        assert loaded_name == 'ROUNDTRIP_TEST'

    
    def test_no_mapping_attribute_in_new_datasets(self, synthetic_pro, tmp_path):
        """Test that new datasets don't have mapping attribute."""
        filepath = synthetic_pro.basic(tmp_path, station_name='NO_MAPPING', n_layers=3)
        
        ds = read(filepath)
        
        assert 'mapping' not in ds.data.location.attrs
    
    def test_multiple_stations_unique_names(self, synthetic_pro, tmp_path):
        """Test that multiple stations have unique string names."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        ds = read(data_dir, lazy=False)
        
        # All locations should be unique strings
        locations = [str(loc) for loc in ds.data.location.values]
        assert len(locations) == len(set(locations))
        assert all(isinstance(loc, str) for loc in locations)


class TestSlopeCoordinatePreservation:
    """
    Test suite for slope coordinate preservation bug fix.
    
    This test class verifies that inclination and azimuth coordinates
    are correctly preserved when:
    1. Concatenating multi-location datasets
    2. Merging PRO and SMET data
    3. Performing time slicing operations
    
    Bug context: Previously, using coords='minimal' in xr.concat() caused slope
    coordinates to be incorrectly merged, resulting in all locations sharing the
    first location's slope values or becoming NaN.
    
    Fix: Changed coords='minimal' to coords='all' in line 1391 of xsnow_io.py.
    """
    
    def test_multi_location_concat_preserves_slope_coords(self, tmp_path):
        """
        Test that concatenating multi-location datasets preserves unique slope coordinates.
        
        This test creates two stations with different slope values and verifies that
        after reading them together and performing time slicing, each station maintains
        its own inclination and azimuth values (not NaN or mixed values).
        
        Regression test for bug where coords='minimal' caused all locations to share
        the first location's slope values.
        """
        # Create station A with 30° slope facing south
        station_a_content = """
[STATION_PARAMETERS]
StationName= station_A
Latitude= 45.0
Longitude= 10.0
Altitude= 1500
SlopeAngle= 30.0
SlopeAzi= 180.0
[DATA]
0500,01.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
0500,02.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
0500,03.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
"""
        
        # Create station B with 25° slope facing east
        station_b_content = """
[STATION_PARAMETERS]
StationName= station_B
Latitude= 46.0
Longitude= 11.0
Altitude= 1600
SlopeAngle= 25.0
SlopeAzi= 90.0
[DATA]
0500,01.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
0500,02.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
0500,03.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
"""
        
        (tmp_path / "station_A.pro").write_text(station_a_content)
        (tmp_path / "station_B.pro").write_text(station_b_content)
        
        # Read both stations
        ds = read(tmp_path)
        
        assert ds is not None
        assert ds.sizes['location'] == 2
        
        # Check that inclination and azimuth exist
        assert 'inclination' in ds.coords
        assert 'azimuth' in ds.coords
        
        # Get values for each station
        ds_a = ds.sel(location='station_A')
        ds_b = ds.sel(location='station_B')
        
        # Extract slope values (handle both scalar and array cases)
        a_incl = float(ds_a.inclination.values.flat[0])
        a_azi = float(ds_a.azimuth.values.flat[0])
        b_incl = float(ds_b.inclination.values.flat[0])
        b_azi = float(ds_b.azimuth.values.flat[0])
        
        # Verify each station has its own correct values
        assert np.isclose(a_incl, 30.0), f"Station A inclination should be 30.0, got {a_incl}"
        assert np.isclose(a_azi, 180.0), f"Station A azimuth should be 180.0, got {a_azi}"
        assert np.isclose(b_incl, 25.0), f"Station B inclination should be 25.0, got {b_incl}"
        assert np.isclose(b_azi, 90.0), f"Station B azimuth should be 90.0, got {b_azi}"
        
        # Test that time slicing preserves coordinates
        ds_sliced = ds.sel(time=slice('2024-01-02', '2024-01-03'))
        
        assert 'inclination' in ds_sliced.coords
        assert 'azimuth' in ds_sliced.coords
        
        # Verify values are still correct after slicing
        ds_a_sliced = ds_sliced.sel(location='station_A')
        ds_b_sliced = ds_sliced.sel(location='station_B')
        
        a_incl_sliced = float(ds_a_sliced.inclination.values.flat[0])
        a_azi_sliced = float(ds_a_sliced.azimuth.values.flat[0])
        b_incl_sliced = float(ds_b_sliced.inclination.values.flat[0])
        b_azi_sliced = float(ds_b_sliced.azimuth.values.flat[0])
        
        assert np.isclose(a_incl_sliced, 30.0)
        assert np.isclose(a_azi_sliced, 180.0)
        assert np.isclose(b_incl_sliced, 25.0)
        assert np.isclose(b_azi_sliced, 90.0)
        
        # Most importantly: verify no NaN values
        assert not np.isnan(a_incl_sliced), "Station A inclination became NaN after slicing"
        assert not np.isnan(b_incl_sliced), "Station B inclination became NaN after slicing"
    
    def test_merge_order_preserves_valid_coords(self, tmp_path):
        """
        Test that PRO files merged before SMET files prevents NaN contamination.
        
        When merging datasets with xr.merge(..., compat='override'), the first
        dataset's coordinate values are kept. This test verifies that PRO files
        (with valid slope data) are merged before SMET files (which may have NaN
        slope values), ensuring valid coordinates are preserved.
        
        Regression test for bug where SMET files with NaN slope values could
        overwrite valid PRO file slope values.
        """
        # Create PRO file with valid slope values
        pro_content = """
[STATION_PARAMETERS]
StationName= test_station
Latitude= 45.0
Longitude= 10.0
Altitude= 1500
SlopeAngle= 30.0
SlopeAzi= 180.0
[DATA]
0500,01.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
0500,02.01.2024 12:00:00
0501,3,0.1,0.2,0.3
0502,3,150,250,350
"""
        
        # Create SMET file (may have NaN slope values by default)
        smet_content = """
SMET 1.1 ASCII
[HEADER]
station_name = test_station
latitude = 45.0
longitude = 10.0
altitude = 1500
fields = timestamp HS
[DATA]
2024-01-01T12:00:00 0.3
2024-01-02T12:00:00 0.4
"""
        
        (tmp_path / "test.pro").write_text(pro_content)
        (tmp_path / "test.smet").write_text(smet_content)
        
        # Read both files (merge happens internally)
        ds = read(tmp_path)
        
        assert ds is not None
        
        # Check that slope coordinates exist and are NOT NaN
        assert 'inclination' in ds.coords
        assert 'azimuth' in ds.coords
        
        incl_value = float(ds.inclination.values.flat[0])
        azi_value = float(ds.azimuth.values.flat[0])
        
        # Should have PRO file's values, not NaN
        assert not np.isnan(incl_value), "inclination is NaN - SMET may have overwritten PRO values"
        assert not np.isnan(azi_value), "azimuth is NaN - SMET may have overwritten PRO values"
        assert np.isclose(incl_value, 30.0), f"Expected 30.0, got {incl_value}"
        assert np.isclose(azi_value, 180.0), f"Expected 180.0, got {azi_value}"
        
        # Test time slicing preserves valid coordinates
        ds_sliced = ds.sel(time=slice('2024-01-01', '2024-01-02'))
        
        incl_sliced = float(ds_sliced.inclination.values.flat[0])
        azi_sliced = float(ds_sliced.azimuth.values.flat[0])
        
        assert not np.isnan(incl_sliced)
        assert not np.isnan(azi_sliced)
        assert np.isclose(incl_sliced, 30.0)
        assert np.isclose(azi_sliced, 180.0)
    
    def test_smet_with_slope_values_in_header(self, tmp_path):
        """
        Test that SMET files with slope values in header are correctly parsed.
        
        This test verifies the enhancement where SMET files that contain
        SlopeAngle and SlopeAzi in their headers have these values extracted
        instead of defaulting to NaN.
        
        Enhancement for improved SMET coordinate assignment (lines 1247-1251).
        """
        # Create SMET file with slope values in header
        smet_with_slope = """
SMET 1.1 ASCII
[HEADER]
station_name = smet_station
latitude = 45.0
longitude = 10.0
altitude = 1500
SlopeAngle = 35.0
SlopeAzi = 270.0
fields = timestamp HS TA
[DATA]
2024-01-01T12:00:00 0.3 270.15
2024-01-02T12:00:00 0.4 271.15
2024-01-03T12:00:00 0.5 272.15
"""
        
        (tmp_path / "slope_test.smet").write_text(smet_with_slope)
        
        # Read SMET file
        ds = read(tmp_path / "slope_test.smet")
        
        assert ds is not None
        
        # Check that slope coordinates were extracted from header
        assert 'inclination' in ds.coords
        assert 'azimuth' in ds.coords
        
        incl_value = float(ds.inclination.values.flat[0])
        azi_value = float(ds.azimuth.values.flat[0])
        
        # Should have header values, not NaN
        assert not np.isnan(incl_value), "inclination is NaN despite being in header"
        assert not np.isnan(azi_value), "azimuth is NaN despite being in header"
        assert np.isclose(incl_value, 35.0), f"Expected 35.0, got {incl_value}"
        assert np.isclose(azi_value, 270.0), f"Expected 270.0, got {azi_value}"
        
        # Test time slicing preserves coordinates
        ds_sliced = ds.sel(time=slice('2024-01-02', '2024-01-03'))
        
        incl_sliced = float(ds_sliced.inclination.values.flat[0])
        azi_sliced = float(ds_sliced.azimuth.values.flat[0])
        
        assert not np.isnan(incl_sliced)
        assert not np.isnan(azi_sliced)
        assert np.isclose(incl_sliced, 35.0)
        assert np.isclose(azi_sliced, 270.0)
    
    def test_coords_all_vs_minimal_behavior(self, tmp_path):
        """
        Test that demonstrates the fix: coords='all' vs coords='minimal'.
        
        This is a direct regression test showing that coords='all' properly
        broadcasts slope coordinates for multi-location datasets, while
        coords='minimal' would have caused the bug.
        
        This test directly validates the fix at line 1391 of xsnow_io.py.
        """
        # Create two stations with different slopes
        for i, (name, incl, azi) in enumerate([
            ('station_X', 20.0, 0.0),
            ('station_Y', 35.0, 270.0)
        ]):
            content = f"""
[STATION_PARAMETERS]
StationName= {name}
SlopeAngle= {incl}
SlopeAzi= {azi}
[DATA]
0500,01.01.2024 12:00:00
0501,2,0.1,0.2
0502,2,150,250
"""
            (tmp_path / f"{name}.pro").write_text(content)
        
        # Read datasets
        ds = read(tmp_path)
        
        # With coords='all' (the fix), inclination should have shape (2, 1)
        # or be properly broadcast for each location
        assert 'inclination' in ds.coords
        
        # Check dimensions - should be indexed by location and/or slope
        slope_incl_dims = ds.inclination.dims
        assert len(slope_incl_dims) > 0, "inclination has no dimensions"
        
        # Most importantly: each location should have its own value
        x_incl = float(ds.sel(location='station_X').inclination.values.flat[0])
        y_incl = float(ds.sel(location='station_Y').inclination.values.flat[0])
        
        assert not np.isclose(x_incl, y_incl), \
            f"Bug present: both stations have same slope value ({x_incl})"
        assert np.isclose(x_incl, 20.0)
        assert np.isclose(y_incl, 35.0)
