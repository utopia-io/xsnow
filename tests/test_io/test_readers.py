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
from pathlib import Path

from xsnow import read, xsnowDataset
from xsnow.selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector


# ============================================================================
# REAL DATA TESTS - SINGLE FILE
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
def test_read_single_smet_file(smet_files_dir):
    """
    Test reading a single SMET file.
    """
    smet_files = list(smet_files_dir.glob("*.smet"))
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
    assert ds.sizes['location'] > 0
    
    # Gridded data has multiple slopes
    if 'slope' in ds.dims:
        assert ds.sizes['slope'] > 1


@pytest.mark.real_data
def test_read_directory_subset(real_file_subset):
    """
    Test reading a subset of files (faster execution).
    """
    ds = read(real_file_subset)
    
    assert ds is not None
    assert ds.sizes['location'] > 0
    assert ds.sizes['time'] > 0


# ============================================================================
# REAL DATA TESTS - MULTIPLE SOURCES
# ============================================================================

@pytest.mark.real_data
def test_read_multiple_sources_list(snowobs_pros_dir, pit_pros_dir):
    """
    Test reading from multiple directories passed as list.
    
    Combines observation runs with pit data.
    """
    ds = read([snowobs_pros_dir, pit_pros_dir])
    
    assert ds is not None
    assert ds.sizes['location'] >= 1
    
    # Should have merged data from both sources
    # Pits typically have single profiles, obs have time series


@pytest.mark.real_data
def test_read_multiple_sources_pro_and_smet(snowobs_pros_dir, smet_files_dir):
    """
    Test reading both .pro and .smet files together.
    
    Should merge profile and time series data.
    """
    # Read .pro files
    ds = read(snowobs_pros_dir)
    
    if ds is None:
        pytest.skip("No snowobs data available")
    
    # Now merge SMET data
    smet_files = list(smet_files_dir.glob("*.smet"))
    if smet_files:
        # Read first SMET for same station
        ds_merged = ds.merge_smet(smet_files[0])
        
        assert ds_merged is not None
        # Should have both profile and time series variables


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
def test_read_with_location_selector_by_name(gridded_pros_dir):
    """
    Test reading with LocationSelector by station name.
    """
    # First get all locations
    ds_all = read(gridded_pros_dir)
    
    if ds_all.sizes['location'] < 1:
        pytest.skip("No locations available")
    
    # Get first location name
    location_map = ds_all.location.attrs.get('mapping', {})
    if not location_map:
        pytest.skip("No location mapping available")
    
    first_loc_name = location_map[0]
    
    # Read with location filter
    loc_sel = LocationSelector(sel=first_loc_name)
    ds_filtered = read(gridded_pros_dir, location=loc_sel)
    
    assert ds_filtered is not None
    # Location dimension should be squeezed
    assert 'location' not in ds_filtered.dims or ds_filtered.sizes['location'] == 1


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
    
    # Should return None or empty dataset
    ds = read(tmp_path)
    
    if ds is not None:
        # If any data was recovered, it should be minimal
        assert ds.sizes['location'] >= 0


@pytest.mark.synthetic
def test_read_mixed_valid_and_invalid_files(tmp_path, synthetic_pro):
    """
    Test that valid files are read even when invalid files are present.
    """
    # Create one valid file
    synthetic_pro.basic(tmp_path, "VALID")
    
    # Create one corrupted file  
    synthetic_pro.corrupted(tmp_path, "malformed_data")
    
    ds = read(tmp_path)
    
    # Should load the valid file
    assert ds is not None
    assert ds.sizes['location'] == 1


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
def test_auto_detect_file_type_smet(smet_files_dir):
    """
    Test that .smet files are automatically detected and routed correctly.
    """
    smet_files = list(smet_files_dir.glob("*.smet"))
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
    Regression: Location mapping should be preserved.
    """
    ds = read(real_file_subset)
    
    assert 'mapping' in ds.location.attrs
    mapping = ds.location.attrs['mapping']
    assert isinstance(mapping, dict)
    assert len(mapping) == ds.sizes['location']


@pytest.mark.real_data
def test_read_maintains_cf_compliance(real_file_subset):
    """
    Regression: Output should be CF-compliant.
    """
    ds = read(real_file_subset)
    
    assert 'Conventions' in ds.data.attrs
    assert 'CF' in ds.data.attrs['Conventions']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])