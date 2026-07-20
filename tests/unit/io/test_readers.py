# tests/io/test_readers.py
"""
Tests for the read() function and file discovery.

Focus: File discovery, routing, selector application, error handling.

Strategy:
  - Use REAL data for primary functionality tests
  - Use SYNTHETIC data for error conditions and edge cases
  - Test all entry points and parameter combinations
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from xsnow import read, xsnowDataset
from xsnow.selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector


# ============================================================================
# REAL DATA TESTS - SINGLE SOURCE
# ============================================================================

@pytest.mark.real_data
def test_read_single_pro_file(sample_pro_file):
    """
    Primary test: Read a single .pro file.
    
    Validates the most basic use case.
    """
    ds = read(sample_pro_file)
    
    assert ds is not None
    assert isinstance(ds, xsnowDataset)
    assert ds.sizes['location'] == 1
    assert ds.sizes['time'] > 0
    assert 'height' in ds.data or 'density' in ds.data


@pytest.mark.real_data
def test_read_single_smet_file(snowobs_smets_dir_nowcast):
    """
    Test reading a single SMET file.
    """
    smet_files = list(snowobs_smets_dir_nowcast.glob("*.smet"))
    if not smet_files:
        pytest.skip("No SMET files available")
    
    ds = read(smet_files[0])
    
    assert ds is not None
    assert isinstance(ds, xsnowDataset)
    assert 'time' in ds.dims


# ============================================================================
# REAL DATA TESTS - DIRECTORY
# ============================================================================

@pytest.mark.real_data
def test_read_directory_gridded(gridded_pros_dir):
    """
    Primary test: Read entire directory of gridded data.
    
    Tests multi-file processing and merging.
    """
    ds = read(gridded_pros_dir)

    assert ds is not None
    assert isinstance(ds, xsnowDataset)
    assert 'time' in ds.sizes
    assert 'timezone' in ds.time.attrs
    assert 'crs' in ds.attrs

    assert ds.sizes['location'] > 1
    assert ds.sizes['location'] == ds.latitude.size == ds.longitude.size == ds.altitude.size
    assert ds.sizes['slope'] == 5
    assert ds.azimuth.dims == ('slope',) == ds.inclination.dims
    assert ds.sizes['slope'] == ds.azimuth.size == ds.inclination.size

    assert ds.latitude.notnull().any()
    assert ds.longitude.notnull().any()
    assert ds.altitude.notnull().any()
    assert ds.azimuth.notnull().any()
    assert ds.inclination.notnull().any()
    assert 'z' in ds.coords

    assert 'density' in ds.data_vars
    assert ds.density.dims == ('location', 'time', 'slope', 'realization', 'layer')
    assert 'TA' in ds.data_vars
    assert ds.TA.dims == ('location', 'time', 'slope', 'realization')


@pytest.mark.real_data
def test_read_directory_subset(real_file_subset):
    """
    Test reading a subset of files (faster execution).
    """
    ds = read(real_file_subset)
    
    assert ds is not None
    assert ds.sizes['location'] > 0
    assert ds.sizes['time'] > 0


@pytest.mark.real_data
def test_read_smets_dir_snowobs(snowobs_smets_dir_nowcast):
    """
    Test reading directory of SMET files.
    """
    ds = read(snowobs_smets_dir_nowcast)
    
    assert ds is not None
    assert isinstance(ds, xsnowDataset)
    assert 'time' in ds.dims
    assert 'timezone' in ds.time.attrs
    assert 'crs' in ds.attrs

    assert ds.sizes['location'] > 1
    assert ds.sizes['location'] == ds.latitude.size == ds.longitude.size == ds.altitude.size
    
    assert ds.latitude.notnull().any()
    assert ds.longitude.notnull().any()
    assert ds.altitude.notnull().any()
    assert ds.azimuth.isnull().all()
    assert ds.inclination.isnull().all()

    assert 'TA' in ds.data_vars
    assert ds.TA.dims == ('location', 'time', 'slope', 'realization')


@pytest.mark.real_data
def test_read_normalize_slope_dim_smet_dir(snowobs_smets_dir_nowcast):
    """
    Test reading multiple SMET files and adjust slope normalization behavior
    The difference to PRO files is, that SMET files have no inclination/azimuth info!
    """
    ds = read(snowobs_smets_dir_nowcast, norm_slopes=True)
    assert ds.azimuth.dims == ('slope',) == ds.inclination.dims
    assert ds.azimuth.isnull().all()
    assert ds.inclination.isnull().all()
    
    ds = read(snowobs_smets_dir_nowcast, norm_slopes=False)
    assert ds.azimuth.dims == ('location', 'slope',) == ds.inclination.dims
    assert ds.azimuth.isnull().all()
    assert ds.inclination.isnull().all()
    
    ds = read(snowobs_smets_dir_nowcast, norm_slopes='auto')
    assert ds.azimuth.dims == ('slope',) == ds.inclination.dims
    assert ds.azimuth.isnull().all()
    assert ds.inclination.isnull().all()


@pytest.mark.real_data
def test_read_normalize_slope_dim_pro_dir(snowobs_pros_dir):
    """
    Test reading dir of .pro and .smet files and adjust slope normalization behavior
    """
    ds = read(snowobs_pros_dir, norm_slopes=True)
    assert ds.azimuth.dims == ('slope',) == ds.inclination.dims
    assert ds.azimuth.notnull().any()
    assert ds.inclination.notnull().any()
    
    ds = read(snowobs_pros_dir, norm_slopes=False)
    assert ds.azimuth.dims == ('location', 'slope',) == ds.inclination.dims
    assert ds.azimuth.notnull().any()
    assert ds.inclination.notnull().any()
    
    ds = read(snowobs_pros_dir, norm_slopes='auto')
    assert ds.azimuth.dims == ('location', 'slope',) == ds.inclination.dims
    assert ds.azimuth.notnull().any()
    assert ds.inclination.notnull().any()


# ============================================================================
# REAL DATA TESTS - MULTIPLE SOURCES
# ============================================================================

@pytest.mark.real_data
def test_read_multiple_sources_list(snowobs_pros_dir, snowobs_pits_dir):
    """
    Test reading from multiple directories passed as list.
    
    Combines observation runs with pit data.
    """
    ds = read([snowobs_pros_dir, snowobs_pits_dir])
    
    assert ds is not None
    assert ds.sizes['location'] >= 1
    
    # Should have merged data from both sources
    # Pits typically have single profiles, obs have time series


@pytest.mark.real_data
def test_read_multiple_pro_and_smet(snowobs_pros_dir):
    """
    Test reading both .pro and .smet files together.
    
    Should merge profile and time series data.
    """
    # Read directory of .pro and .smet files:
    ds = read(snowobs_pros_dir)

    assert 'time' in ds.coords
    assert 'timezone' in ds.time.attrs
    assert 'density' in ds.data_vars
    assert 'HS' in ds.data_vars       # from .pro file
    assert 'ski_pen' in ds.data_vars  # from .smet file


@pytest.mark.real_data
def test_read_conflicting_pro_smet_dirs(snowobs_pros_dir, snowobs_smets_dir_nowcast):
    """
    Test reading two directories with different timezones, should raise
    """
    with pytest.raises(ValueError):
        ds = read([snowobs_pros_dir, snowobs_smets_dir_nowcast])


@pytest.mark.real_data
def test_merge_conflicting_smet_dir(snowobs_pros_dir, snowobs_smets_dir_nowcast):
    """
    Read a pro dir, then try merge_smet with a conflicting timezone, should raise
    """
    ds = read(snowobs_pros_dir)
    with pytest.raises(ValueError):
        ds.merge_smet(snowobs_smets_dir_nowcast)


@pytest.mark.real_data
def test_merge_smets_snowobs(snowobs_pros_dir):
    """
    Read all pros from dir, then merge_smet
    """
    pro_files = list(snowobs_pros_dir.glob("*.pro"))
    smet_files = list(snowobs_pros_dir.glob("*.smet"))
    ds = read(pro_files)
    ds = ds.merge_smet(smet_files, var_prefix='input_')

    assert 'time' in ds.sizes
    assert 'timezone' in ds.time.attrs
    assert 'crs' in ds.attrs

    assert ds.sizes['location'] > 1
    assert ds.sizes['location'] == ds.latitude.size == ds.longitude.size == ds.altitude.size
    assert ds.sizes['slope'] >= 1

    assert ds.latitude.notnull().any()
    assert ds.longitude.notnull().any()
    assert ds.altitude.notnull().any()
    assert ds.azimuth.notnull().any()
    assert ds.inclination.notnull().any()
    assert 'z' in ds.coords

    assert 'density' in ds.data_vars
    assert ds.density.dims == ('location', 'time', 'slope', 'realization', 'layer')
    assert 'input_ski_pen' in ds.data_vars
    assert ds.input_ski_pen.dims == ('location', 'time', 'slope', 'realization')



# ============================================================================
# REAL DATA TESTS - NETCDF
# ============================================================================

@pytest.mark.real_data
def test_read_netcdf_file(real_file_subset, tmp_path):
    """
    Test reading from NetCDF file.
    
    Workflow: read → save → read again
    """
    # Read original data
    ds_original = read(real_file_subset)
    
    # Save to NetCDF
    nc_file = tmp_path / "test.nc"
    ds_original.to_netcdf(nc_file)
    
    # Read back
    ds_from_nc = read(nc_file)
    
    assert ds_from_nc is not None
    assert ds_from_nc.sizes == ds_original.sizes


def _write_netcdf_selector_dataset(tmp_path: Path) -> Path:
    """Write a small multi-coordinate NetCDF dataset for selector tests."""
    locations = ["A", "B"]
    times = pd.date_range("2025-01-01", periods=3, freq="h")
    slopes = [0, 1]
    realizations = ["r0", "r1"]
    shape = (len(locations), len(times), len(slopes), len(realizations))
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)

    ds = xr.Dataset(
        data_vars={
            "marker": (
                ("location", "time", "slope", "realization"),
                values,
            ),
        },
        coords={
            "location": locations,
            "time": times,
            "slope": slopes,
            "realization": realizations,
            "latitude": ("location", [46.5, 47.0]),
            "longitude": ("location", [10.5, 11.0]),
            "altitude": ("location", [2500.0, 2600.0]),
            "inclination": ("slope", [0.0, 35.0]),
            "azimuth": ("slope", [0.0, 180.0]),
        },
    )
    nc_file = tmp_path / "selector_source.nc"
    ds.to_netcdf(nc_file)
    return nc_file


def test_read_netcdf_applies_selectors(tmp_path):
    """NetCDF reads should subset before eager loading when selectors are given."""
    nc_file = _write_netcdf_selector_dataset(tmp_path)

    loaded = read(
        nc_file,
        time=slice("2025-01-01T01:00", None),
        location=["B"],
        slope=[1],
        realization=["r1"],
        lazy=False,
    )

    assert loaded.sizes == {
        "location": 1,
        "time": 2,
        "slope": 1,
        "realization": 1,
    }
    assert loaded.location.values.tolist() == ["B"]
    assert loaded.realization.values.tolist() == ["r1"]
    assert loaded.slope.values.tolist() == [1]
    np.testing.assert_array_equal(
        loaded.time.values,
        np.array(
            ["2025-01-01T01:00:00", "2025-01-01T02:00:00"],
            dtype="datetime64[ns]",
        ),
    )

    expected = np.array([[[[19.0]], [[23.0]]]])
    np.testing.assert_allclose(loaded["marker"].values, expected)
    np.testing.assert_allclose(loaded.inclination.values, [35.0])
    np.testing.assert_allclose(loaded.azimuth.values, [180.0])


def test_read_netcdf_lazy_applies_selectors(tmp_path):
    """Lazy NetCDF reads should expose the same selected coordinate domain."""
    nc_file = _write_netcdf_selector_dataset(tmp_path)

    loaded = read(nc_file, location="A", slope=0, realization="r0", lazy=True)

    assert "location" not in loaded.sizes
    assert "slope" not in loaded.sizes
    assert "realization" not in loaded.sizes
    assert loaded.sizes["time"] == 3
    np.testing.assert_allclose(loaded["marker"].values, [0.0, 4.0, 8.0])


# ============================================================================
# REAL DATA TESTS - WITH SELECTORS
# ============================================================================

@pytest.mark.real_data
def test_read_with_time_selector(real_file_subset):
    """
    Test reading with TimeSelector applied at source.
    
    This is more efficient than loading all data then filtering.
    """
    # First load to know time range
    ds_full = read(real_file_subset)
    
    if ds_full.sizes['time'] < 2:
        pytest.skip("Need multiple timesteps for time filtering")
    
    # Get middle time
    mid_idx = ds_full.sizes['time'] // 2
    mid_time = ds_full.time.values[mid_idx]
    
    # Read with time filter
    time_sel = TimeSelector(sel=slice(None, mid_time))
    ds_filtered = read(real_file_subset, time=time_sel)
    
    assert ds_filtered is not None
    assert ds_filtered.sizes['time'] <= ds_full.sizes['time']


@pytest.mark.real_data
def test_read_with_location_selector_by_name(real_file_subset):
    """
    Test location filtering by station name using LocationSelector.
    
    Updated: Now works directly with string-based location coordinates.
    """
    # Read all data
    ds_all = read(real_file_subset)
    
    # Get first location name directly from coordinate
    first_location = str(ds_all.location.values[0])
    
    # Read with location filter
    from xsnow.selectors import LocationSelector
    loc_filter = LocationSelector(sel=first_location)
    ds_filtered = read(real_file_subset, location=loc_filter)
    
    # Should have only the selected location
    if 'location' in ds_filtered.sizes:
        assert ds_filtered.sizes['location'] == 1
        assert str(ds_filtered.location.values[0]) == first_location
    else:
        # Location dimension was squeezed
        assert str(ds_filtered.location.values) == first_location


@pytest.mark.real_data
def test_read_with_slope_selector(gridded_pros_dir):
    """
    Test reading with SlopeSelector.
    
    Gridded data has multiple slopes per station.
    """
    # Read all slopes first
    ds_all = read(gridded_pros_dir)
    
    if 'slope' not in ds_all.dims or ds_all.sizes['slope'] < 2:
        pytest.skip("Need multiple slopes for slope filtering")
    
    # Select first slope
    slope_sel = SlopeSelector(sel=0)
    ds_filtered = read(gridded_pros_dir, slope=slope_sel)
    
    assert ds_filtered is not None
    # Slope dimension should be squeezed
    assert 'slope' not in ds_filtered.dims


@pytest.mark.real_data
def test_read_with_multiple_selectors(real_file_subset):
    """
    Test combining multiple selectors.
    """
    ds_full = read(real_file_subset)
    
    if ds_full.sizes['time'] < 2:
        pytest.skip("Need multiple timesteps")
    
    # Apply time filter with single value (will squeeze time dim)
    time_sel = TimeSelector(sel=ds_full.time.values[0])
    
    ds_filtered = read(real_file_subset, time=time_sel)
    
    assert ds_filtered is not None
    # Single time selection squeezes the time dimension
    if 'time' in ds_filtered.sizes:
        assert ds_filtered.sizes['time'] <= ds_full.sizes['time']
    else:
        # Time dimension was squeezed
        assert 'time' not in ds_filtered.dims

# ============================================================================
# SYNTHETIC DATA TESTS - ERROR HANDLING
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_read_empty_directory(tmp_path):
    """
    Edge case: Empty directory should return None.
    """
    ds = read(tmp_path)
    assert ds is None


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_read_nonexistent_path(tmp_path):
    """
    Edge case: Nonexistent path should return None.
    """
    fake_path = tmp_path / "nonexistent" / "path"
    ds = read(fake_path)
    assert ds is None


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_read_directory_with_only_invalid_files(tmp_path, synthetic_pro):
    """
    Edge case: Directory with only corrupted files.
    """
    # Create multiple corrupted files
    synthetic_pro.corrupted(tmp_path, "malformed_data")
    synthetic_pro.corrupted(tmp_path, "empty")  # FIX: Just pass tmp_path, not a file path
    
    # Should return proper dataset but profile_status <=0
    ds = read(tmp_path)
    
    assert all(dim in ds.dims for dim in ['location', 'time', 'slope', 'realization'])
    assert ds.sizes['location'] >= 0
    assert (ds.profile_status.values.squeeze() <= 0).all()


@pytest.mark.synthetic
def test_read_mixed_files_intricate_profile_status(tmp_path, synthetic_pro):
    """
    Test that a dir of valid/corrupt/now-snow files are being read correctly,
    and that intricate profile_status expectations are met:
    
    VALID profile: [valid, NA] for [t1, t2] --> profile_status [1, 0]
    CORRUPTED profile: [corrupt, NA] for [t1, t2] --> profile_status [-1, 0]
    no_snow profile: [NA, valid/no-snow] for [t1, t2] --> profile_status [0, 1]

    Note that the dataset needs to be created with two timestamps, although the second
    timestamp only originates from the no-snow profile.
    """

    # Create one valid file
    synthetic_pro.basic(tmp_path, "VALID", 3, "01.01.2025 12:00:00")
    # Create one file with no snow
    synthetic_pro.basic(tmp_path, "no_snow", 0, "01.01.2025 13:00:00")
    # Create one corrupted file
    synthetic_pro.corrupted(tmp_path, "malformed_data")
    ds = read(tmp_path)

    assert ds.sizes['location'] == 3
    assert ds.sizes['time'] == 2
    np.testing.assert_array_equal(ds.profile_status.sel(location='VALID').squeeze().values, [1, 0])
    np.testing.assert_array_equal(ds.profile_status.sel(location='CORRUPTED').squeeze().values, [-1, 0])
    np.testing.assert_array_equal(ds.profile_status.sel(location='no_snow').squeeze().values, [0, 1])


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_read_corrupted_file_good_and_bad_data(tmp_path, synthetic_pro):
    """
    Good and bad data for two different times. profile_status should identify.
    """
    corrupted = synthetic_pro.corrupted(tmp_path, "good_and_bad_data")
    xs = read(corrupted)
    assert (xs.profile_status.squeeze().values == [1, -1]).all()


# ============================================================================
# FILE TYPE DETECTION TESTS
# ============================================================================

@pytest.mark.real_data
def test_auto_detect_file_type_pro(sample_pro_file):
    """
    Test that .pro files are automatically detected and routed correctly.
    """
    ds = read(sample_pro_file)
    
    assert ds is not None
    # .pro files have layer dimension
    assert 'layer' in ds.dims


@pytest.mark.real_data  
def test_auto_detect_file_type_smet(snowobs_smets_dir_nowcast):
    """
    Test that .smet files are automatically detected and routed correctly.
    """
    smet_files = list(snowobs_smets_dir_nowcast.glob("*.smet"))
    if not smet_files:
        pytest.skip("No SMET files available")
    
    ds = read(smet_files[0])
    
    assert ds is not None
    # .smet files typically don't have layer dimension
    # (they're time series, not profiles)


@pytest.mark.real_data
def test_auto_detect_file_type_netcdf(real_file_subset, tmp_path):
    """
    Test that .nc files are automatically detected and routed correctly.
    """
    # Create a NetCDF file
    ds_original = read(real_file_subset)
    nc_file = tmp_path / "test.nc"
    ds_original.to_netcdf(nc_file)
    
    # Read it back - should auto-detect NetCDF
    ds = read(nc_file)
    
    assert ds is not None
    assert isinstance(ds, xsnowDataset)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_read_preserves_physical_validity(real_file_subset, assert_physical_validity):
    """
    Integration: Ensure read() produces physically valid data.
    """
    ds = read(real_file_subset)
    
    assert ds is not None
    assert_physical_validity(ds)


@pytest.mark.real_data
def test_read_lazy_vs_eager_consistency(real_file_subset, compare_datasets):
    """
    Integration: Lazy and eager loading should produce same results.
    """
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    
    assert ds_lazy is not None
    assert ds_eager is not None
    
    compare_datasets(ds_lazy, ds_eager, rtol=1e-5)


@pytest.mark.real_data
def test_read_computes_z_coordinate(real_file_subset):
    """
    Integration: Read should compute z-depth coordinate automatically.
    """
    ds = read(real_file_subset)
    
    if ds and 'height' in ds.data:
        # z coordinate should be present
        assert 'z' in ds.data.coords
        
        # z should be negative (below surface)
        z_values = ds.data['z'].values
        valid_z = z_values[~np.isnan(z_values)]
        if len(valid_z) > 0:
            assert np.all(valid_z <= 0)


@pytest.mark.real_data
@pytest.mark.slow
def test_read_large_dataset_performance(gridded_pros_dir, timer):
    """
    Integration: Performance test on full dataset.
    """
    timer.start('read_full')
    ds = read(gridded_pros_dir, n_cpus_use=4)
    timer.stop('read_full')
    
    timer.report()
    
    assert ds is not None
    
    # Should complete in reasonable time
    elapsed = timer.get('read_full')
    if elapsed:
        print(f"\nRead {ds.sizes['location']} locations in {elapsed:.2f}s")


# ============================================================================
# PARAMETER VALIDATION TESTS
# ============================================================================

@pytest.mark.synthetic
def test_read_with_max_layers_override(tmp_path, synthetic_pro):
    """
    Test manual max_layers parameter.
    """
    synthetic_pro.basic(tmp_path, n_layers=3)
    
    # Force max_layers to specific value
    ds = read(tmp_path, max_layers=10)
    
    assert ds is not None
    assert ds.sizes['layer'] == 10


@pytest.mark.synthetic
def test_read_with_remove_soil_parameter(tmp_path, synthetic_pro):
    """
    Test remove_soil parameter.
    """
    synthetic_pro.with_soil_layers(tmp_path, n_snow=3, n_soil=2)
    
    # With soil removal
    ds_no_soil = read(tmp_path, remove_soil=True)
    
    # Without soil removal
    ds_with_soil = read(tmp_path, remove_soil=False)
    
    assert ds_no_soil.sizes['layer'] < ds_with_soil.sizes['layer']


@pytest.mark.synthetic
def test_read_with_add_surface_sh_parameter(tmp_path, synthetic_pro):
    """
    Test add_surface_sh_as_layer parameter.
    """
    synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
    
    # With SH addition
    ds_with_sh = read(tmp_path, add_surface_sh_as_layer=True)
    
    # Without SH addition
    ds_no_sh = read(tmp_path, add_surface_sh_as_layer=False)
    
    # Should have one more layer with SH
    assert ds_with_sh.sizes['layer'] == ds_no_sh.sizes['layer'] + 1


@pytest.mark.synthetic
def test_read_with_n_cpus_parameter(tmp_path, synthetic_pro):
    """
    Test n_cpus_use parameter.
    """
    synthetic_pro.multi_station(tmp_path, n_stations=5)
    
    # Try different CPU counts
    ds_1cpu = read(tmp_path, n_cpus_use=1)
    ds_4cpu = read(tmp_path, n_cpus_use=4)
    
    # Both should work and produce same result
    assert ds_1cpu is not None
    assert ds_4cpu is not None
    assert ds_1cpu.sizes == ds_4cpu.sizes


# ============================================================================
# REGRESSION TESTS
# ============================================================================

@pytest.mark.real_data
def test_read_returns_xsnowdataset_instance(sample_pro_file):
    """
    Regression: read() should always return xsnowDataset, not raw xarray.
    """
    ds = read(sample_pro_file)
    
    assert isinstance(ds, xsnowDataset)
    assert not isinstance(ds, type(ds.data))  # Not raw xr.Dataset


@pytest.mark.real_data
def test_read_preserves_location_mapping(real_file_subset):
    """
    Test: Location coordinate should be meaningful string values.
    
    Updated from legacy behavior: Previously used integer indices with a 
    'mapping' attribute. Now uses string station names directly as coordinate
    values, which is more intuitive and follows xarray best practices.
    """
    ds = read(real_file_subset)
    
    # Verify location is a string-based coordinate
    assert 'location' in ds.coords
    
    # Location values should be strings (station names)
    location_values = ds.location.values
    assert location_values.dtype.kind in ('U', 'S', 'O')  # Unicode, bytes, or object (strings)
    
    # Should have at least one location
    assert len(location_values) > 0
    
    # Each location should be a non-empty string
    for loc in location_values:
        loc_str = str(loc)
        assert loc_str  # Not empty
        assert loc_str != 'nan'  # Not a placeholder
    
    # The old 'mapping' attribute is no longer needed since
    # location values are already meaningful strings
    # (this is the intentional new behavior)


@pytest.mark.real_data
def test_read_maintains_cf_compliance(real_file_subset):
    """
    Regression: Output should be CF-compliant.
    """
    ds = read(real_file_subset)
    
    assert 'Conventions' in ds.data.attrs
    assert 'CF' in ds.data.attrs['Conventions']

@pytest.mark.synthetic
class TestStringCoordinateReading:
    """Test that readers create string location coordinates."""
    
    def test_single_file_has_string_location(self, synthetic_pro, tmp_path):
        """Test reading single .pro creates string location."""
        from xsnow import read
        
        filepath = synthetic_pro.basic(tmp_path, station_name='VIR1A', n_layers=3)
        ds = read(filepath)
        
        # Location should be string type
        assert ds.data.location.dtype.kind in ('U', 'S', 'O')
        assert ds.data.location.values[0] == 'VIR1A'
    
    def test_multiple_files_have_string_locations(self, synthetic_pro, tmp_path):
        """Test multiple files create string location coords."""
        from xsnow import read
        
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=3)
        ds = read(data_dir, lazy=False)
        
        # All locations should be strings
        assert ds.data.location.dtype.kind in ('U', 'S', 'O')
        assert all(isinstance(str(loc), str) for loc in ds.data.location.values)
    
    def test_no_mapping_attribute(self, synthetic_pro, tmp_path):
        """Test that mapping attribute is not present."""
        from xsnow import read
        
        filepath = synthetic_pro.basic(tmp_path, station_name='TEST', n_layers=3)
        ds = read(filepath)
        
        # Should NOT have mapping attribute
        assert 'mapping' not in ds.data.location.attrs
        
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
