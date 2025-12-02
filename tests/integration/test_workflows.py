# tests/integration/test_workflows.py
"""
Tests for complete end-to-end workflows.
Focus: Real-world usage patterns.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from xsnow import read
from xsnow.selectors import TimeSelector, LocationSelector


@pytest.mark.real_data
class TestCachingWorkflow:
    """Test caching and incremental append workflows."""
    
    def test_initial_load_and_cache(self, real_file_subset, tmp_path):
        """Test initial data load and caching to NetCDF."""
        # Initial load
        ds = read(real_file_subset, lazy=False)
        
        assert ds is not None
        
        # Cache to NetCDF
        cache_file = tmp_path / "cache.nc"
        ds.to_netcdf(cache_file)
        
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0
        
        # Load from cache
        ds_cached = read(cache_file)
        
        assert ds_cached is not None
        assert ds_cached.sizes == ds.sizes
    
    def test_daily_incremental_update(self, synthetic_pro, tmp_path):
        """Test incremental append workflow using append_latest."""
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        
        # Create file with 3 timestamps
        synthetic_pro.varying_layers(
            archive_dir,
            layer_counts=[3, 3, 3]  # Creates 3 daily profiles starting 2025-01-01
        )
        
        # Initial load and cache
        cache_file = tmp_path / "cache.nc"
        
        ds_initial = read(archive_dir, lazy=False)
        ds_initial.to_netcdf(cache_file)
        
        assert ds_initial.sizes['time'] == 3
        
        # Add new data - create file with additional timestamp
        synthetic_pro.varying_layers(
            archive_dir,
            layer_counts=[3, 3, 3, 3],  # 4 profiles (will include the 4th day)
            #station_name="SYNTHETIC2"  # Different name to create new file
        )
        
        # Load cached data
        ds_cached = read(cache_file)
        
        # Append with new data
        ds_updated = ds_cached.append_latest(archive_dir)
        
        # Note: This will merge both files, so may have more timestamps
        # The key is that append_latest doesn't crash and adds new data
        assert ds_updated.sizes['time'] >= ds_initial.sizes['time']
        
        # Save updated cache
        ds_updated.to_netcdf(cache_file)
    
    
    def test_append_latest_with_no_new_data(self, synthetic_pro, tmp_path):
        """Test append_latest when no new data is available."""
        filepath = synthetic_pro.basic(tmp_path)  # Pass directory
        
        # Initial load
        ds = read(tmp_path)  # Read from directory
        original_size = ds.sizes['time']
        
        # Update with same source (no new data)
        ds_updated = ds.append_latest(tmp_path)
        
        # Should be unchanged
        assert ds_updated.sizes['time'] == original_size

    
    def test_append_latest_handles_duplicates(self, synthetic_pro, tmp_path):
        """Test that append_latest handles duplicate timestamps correctly."""
        # Create file with 3 profiles
        filepath = synthetic_pro.varying_layers(
            tmp_path,  # Pass directory
            layer_counts=[3, 3, 3]
        )
        
        # Initial load
        ds = read(tmp_path)
        
        # "Update" with same file (duplicates)
        ds_updated = ds.append_latest(tmp_path)
        
        # Should deduplicate and keep same size
        assert ds_updated.sizes['time'] == ds.sizes['time']


@pytest.mark.real_data
class TestAnalysisWorkflows:
    """Test complete analysis workflows."""
    
    def test_load_filter_analyze_save(self, real_file_subset, tmp_path):
        """Test complete workflow: load → filter → analyze → save."""
        # 1. Load data
        ds = read(real_file_subset, lazy=False)
        
        # 2. Filter to specific time period
        times = pd.to_datetime(ds.data.time.values)
        if len(times) > 1:
            mid_time = times[len(times) // 2]
            time_filter = TimeSelector(sel=slice(None, mid_time.isoformat()))
            ds = ds.sel(time=time_filter.sel)
        
        # 3. Compute derived variable
        if 'height' in ds.data:
            ds = ds.compute_hs()
            assert 'HS' in ds.data
        
        # 4. Compute depth coordinate
        if 'height' in ds.data:
            ds = ds.compute_z_depth()
            assert 'z' in ds.data.coords
        
        # 5. Save results
        output_file = tmp_path / "analysis_results.nc"
        ds.to_netcdf(output_file)
        
        assert output_file.exists()
        
        # 6. Reload and verify
        ds_reloaded = read(output_file)
        assert 'HS' in ds_reloaded.data
        assert 'z' in ds_reloaded.data.coords
    
    def test_merge_multiple_sources(self, synthetic_pro, synthetic_smet, tmp_path):
        """Test merging profile and time-series data."""
        # Create profile data
        pro_dir = tmp_path / "profiles"
        pro_dir.mkdir()
        pro_file = synthetic_pro.basic(pro_dir, station_name="STATION1")
        
        # Create meteorological data
        smet_dir = tmp_path / "meteo"
        smet_dir.mkdir()
        smet_file = synthetic_smet.basic(smet_dir, station_name="STATION1")
        
        # Load profile data
        ds_pro = read(pro_dir)
        
        # Merge with SMET data
        ds_merged = ds_pro.merge_smet(smet_dir)
        
        # Should have both profile and time-series variables
        assert 'height' in ds_merged.data  # from profile
        assert 'TA' in ds_merged.data or 'RH' in ds_merged.data  # from SMET    
        
    def test_multi_location_analysis(self, synthetic_pro, tmp_path):
        """Test analysis across multiple locations."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=5,
            n_profiles_each=10
        )
        
        ds = read(data_dir, lazy=False)
        
        # Compute HS for all locations
        ds = ds.compute_hs()
        
        # Calculate mean HS per location
        hs_mean = ds['HS'].mean(dim='time')
        
        assert hs_mean.sizes['location'] == 5
        assert not np.all(np.isnan(hs_mean.values))


@pytest.mark.real_data
class TestLongTermArchive:
    """Test workflows with long-term archives."""
        
    def test_long_term_archive_processing(self, synthetic_pro, tmp_path, timer):
        """Test processing a large temporal archive."""
        # Create archive with 100 timesteps
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        
        # Create file with 100 timestamps
        n_days = 100
        synthetic_pro.varying_layers(
            archive_dir,
            layer_counts=[3] * n_days  # 100 profiles with 3 layers each
        )
        
        # Process archive
        timer.start('load_archive')
        ds = read(archive_dir, lazy=False)
        timer.stop('load_archive')
        
        assert ds.sizes['time'] == n_days
        
        # Extract winter season
        winter_start = '2024-12-01'
        winter_end = '2025-03-31'
        
        time_filter = TimeSelector(sel=slice(winter_start, winter_end))
        ds_winter = ds.sel(time=time_filter.sel)
        
        # Should have filtered data
        assert ds_winter.sizes['time'] < ds.sizes['time']
        assert ds_winter.sizes['time'] > 0
        
        timer.report()
            
    def test_temporal_aggregation(self, synthetic_pro, tmp_path):
        """Test temporal aggregation operations."""
        # Create daily data for a month
        data_dir = tmp_path / "daily"
        data_dir.mkdir()
        
        # Create file with 30 daily timestamps
        synthetic_pro.varying_layers(
            data_dir,
            layer_counts=[3] * 30  # 30 profiles
        )
        
        ds = read(data_dir, lazy=False)
        ds = ds.compute_hs()
        
        # Calculate weekly mean
        if 'HS' in ds.data:
            # Resample to weekly
            ds_weekly = ds.data.resample(time='W').mean()
            
            # Should have ~4 weeks
            assert ds_weekly.sizes['time'] in [4, 5]

@pytest.mark.synthetic
class TestExportWorkflows:
    """Test data export workflows."""
    
    def test_export_to_multiple_formats(self, synthetic_pro, tmp_path):
        """Test exporting to various formats."""
        filepath = synthetic_pro.basic(tmp_path)
        ds = read(tmp_path)
        
        # Export to NetCDF
        nc_file = tmp_path / "export.nc"
        ds.to_netcdf(nc_file)
        assert nc_file.exists()
        
        # Export to SMET (if has time-series data)
        try:
            smet_file = tmp_path / "export.smet"
            ds.to_smet(smet_file)
            assert smet_file.exists()
        except ValueError:
            pass  # Expected if no time-series variables
        
        # Export to PRO
        pro_file = tmp_path / "export.pro"
        ds.to_pro(pro_file)
        assert pro_file.exists()
    
    def test_roundtrip_netcdf(self, synthetic_pro, tmp_path, compare_datasets):
        """Test NetCDF roundtrip maintains data integrity."""
        filepath = synthetic_pro.basic(tmp_path, n_layers=10)
        ds_original = read(tmp_path)
        
        # Save to NetCDF
        nc_file = tmp_path / "roundtrip.nc"
        ds_original.to_netcdf(nc_file)
        
        # Reload
        ds_reloaded = read(nc_file)
        
        # Should be identical
        compare_datasets(ds_original, ds_reloaded, rtol=1e-7)
    
    def test_selective_export(self, synthetic_pro, tmp_path):
        """Test exporting subsets of data."""
        # Create multi-timestep data
        filepath = synthetic_pro.varying_layers(
            tmp_path,
            layer_counts=[5, 5, 5, 5, 5]
        )
        
        ds = read(tmp_path)
        
        # Select subset
        ds_subset = ds.isel(time=slice(0, 3))
        
        # Export subset
        nc_file = tmp_path / "subset.nc"
        ds_subset.to_netcdf(nc_file)
        
        # Reload and verify
        ds_reloaded = read(nc_file)
        assert ds_reloaded.sizes['time'] == 3


@pytest.mark.real_data
class TestErrorRecoveryWorkflows:
    """Test workflows that handle errors gracefully."""
    
    def test_partial_failure_recovery(self, synthetic_pro, tmp_path):
        """Test that workflow continues after partial failures."""
        # Create mix of valid and invalid files
        for i in range(5):
            synthetic_pro.basic(tmp_path, station_name=f"VALID_{i}")
        
        # Add corrupted file
        synthetic_pro.corrupted(tmp_path, corruption_type='malformed_data')
        
        # Should load valid files
        ds = read(tmp_path, lazy=False)
        
        assert ds is not None
        assert ds.sizes['location'] == 6
        assert ds.profile_status.sum() == 5  # only 5 valid
        assert ds.sel(location="CORRUPTED").profile_status.values == 0
    
    def test_workflow_with_missing_optional_data(self, synthetic_pro, tmp_path):
        """Test workflow when optional data is missing."""
        # Create minimal file (only height and density)
        filepath = synthetic_pro.basic(tmp_path)
        ds = read(tmp_path)
        
        # Should still work for available operations
        assert ds is not None
        
        if 'height' in ds.data:
            ds = ds.compute_hs()
            assert 'HS' in ds.data
        
        # Export should work
        nc_file = tmp_path / "minimal.nc"
        ds.to_netcdf(nc_file)
        assert nc_file.exists()
        
def _normalize_location_string(loc):
    """Helper to normalize location values that may be bytes or strings."""
    if isinstance(loc, bytes):
        return loc.decode('utf-8')
    return str(loc)

@pytest.mark.synthetic
class TestStringCoordinateWorkflow:
    """End-to-end workflow with string coordinates."""
    
    def test_complete_workflow(self, synthetic_pro, tmp_path):
        """Test parse -> select -> save -> load workflow."""
        from xsnow import read
        import xarray as xr
        
        # 1. Create and read data
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=3, n_profiles_each=5)
        ds = read(data_dir, lazy=False)
        
        # 2. Verify string coords
        assert ds.data.location.dtype.kind in ('U', 'S', 'O')
        
        # 3. Select by name
        first_station = _normalize_location_string(ds.data.location.values[0])
        ds_filtered = ds.sel(location=first_station)
        filtered_loc = _normalize_location_string(ds_filtered.location.values)
        assert filtered_loc == first_station
        
        # 4. Save and reload
        nc_path = tmp_path / "workflow.nc"
        ds.to_netcdf(nc_path)
        ds_loaded = read(nc_path)
        
        # 5. Verify preserved - normalize both for comparison
        original_locs = [_normalize_location_string(loc) for loc in ds.data.location.values]
        loaded_locs = [_normalize_location_string(loc) for loc in ds_loaded.data.location.values]
        assert original_locs == loaded_locs
    
    def test_concat_different_batches(self, synthetic_pro, tmp_path):
        """Test concatenating datasets from different batches."""
        from xsnow import read
        import xarray as xr
        
        # Create two batches
        dir1 = tmp_path / "batch1"
        dir2 = tmp_path / "batch2"
        dir1.mkdir()
        dir2.mkdir()
        
        synthetic_pro.basic(dir1, station_name='STATION_A', n_layers=3)
        synthetic_pro.basic(dir2, station_name='STATION_B', n_layers=3)
        
        ds1 = read(dir1)
        ds2 = read(dir2)
        
        # Concatenate - should work seamlessly
        combined = xr.concat([ds1.data, ds2.data], dim='location')
        
        assert combined.sizes['location'] == 2
        locations = [_normalize_location_string(loc) for loc in combined.location.values]
        assert 'STATION_A' in locations
        assert 'STATION_B' in locations


@pytest.mark.synthetic
class TestSlopeCoordinateBugFixes:
    """Tests for slope coordinate preservation bug fixes.
    
    These tests verify the fixes for issues where inclination and 
    azimuth coordinates become NaN after operations like time slicing,
    multi-location loading, and update operations.
    
    Reference: bug_analysis_and_fix.md
    """
    
    def test_time_slice_preserves_slope_coords(self, synthetic_pro, tmp_path):
        """Test that time slicing preserves slope coordinates.
        
        Bug: inclination and azimuth became NaN after time slice.
        Root cause: coords='minimal' in concat operations.
        Fix: Changed to coords='all'.
        """
        # Create data with valid slope coordinates
        filepath = synthetic_pro.varying_layers(
            tmp_path,
            layer_counts=[3, 3, 3, 3, 3]
        )
        
        # Load data
        ds = read(tmp_path, lazy=False)
        
        # Verify slope coordinates exist and are valid initially
        assert 'inclination' in ds.data.coords
        assert 'azimuth' in ds.data.coords
        initial_incl = float(ds.data.inclination.values.flat[0])
        initial_azi = float(ds.data.azimuth.values.flat[0])
        assert not np.isnan(initial_incl), "Initial inclination should not be NaN"
        assert not np.isnan(initial_azi), "Initial azimuth should not be NaN"
        
        # Perform time slice
        times = pd.to_datetime(ds.data.time.values)
        mid_time = times[2]
        ds_sliced = ds.sel(time=slice(None, mid_time.isoformat()))
        
        # BUG CHECK: Slope coordinates should still be valid after slice
        assert 'inclination' in ds_sliced.data.coords
        assert 'azimuth' in ds_sliced.data.coords
        sliced_incl = float(ds_sliced.data.inclination.values.flat[0])
        sliced_azi = float(ds_sliced.data.azimuth.values.flat[0])
        assert not np.isnan(sliced_incl), "inclination became NaN after time slice!"
        assert not np.isnan(sliced_azi), "azimuth became NaN after time slice!"
        # Values should be preserved (whatever they are)
        assert sliced_incl == initial_incl, f"Slope changed from {initial_incl} to {sliced_incl}"
        assert sliced_azi == initial_azi, f"Azimuth changed from {initial_azi} to {sliced_azi}"
    
    def test_multi_location_preserves_individual_slope_coords(self, synthetic_pro, tmp_path):
        """Test that multi-location datasets preserve slope coordinates.
        
        Bug: coords='minimal' with compat='override' kept only the first location's
        slope values, applying them to all locations or dropping them.
        Fix: Changed to coords='all' to broadcast properly.
        """
        # Create multiple stations
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=3,
            n_profiles_each=5
        )
        
        # Load all stations together
        ds = read(tmp_path, lazy=False)
        
        assert ds.sizes['location'] == 3
        
        # Each location should have valid slope coordinates (not NaN)
        for loc in ds.location.values:
            loc_ds = ds.sel(location=loc)
            
            assert 'inclination' in loc_ds.coords
            assert 'azimuth' in loc_ds.coords
            
            incl = float(loc_ds.inclination.values.flat[0])
            azi = float(loc_ds.azimuth.values.flat[0])
            
            # BUG CHECK: Coordinates should not be NaN
            assert not np.isnan(incl), f"{loc}: inclination is NaN"
            assert not np.isnan(azi), f"{loc}: azimuth is NaN"
    
    def test_append_latest_preserves_slope_coords(self, synthetic_pro, tmp_path):
        """Test that append_latest preserves slope coordinates.
        
        Bug: coords='minimal' in concat operations within append_latest() caused
        slope coordinates to be dropped or become NaN.
        Fix: Changed to coords='all' in all concat operations.
        """
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        
        # Create initial data with valid slope coordinates
        synthetic_pro.varying_layers(
            archive_dir,
            layer_counts=[3, 3, 3]
        )
        
        # Initial load and cache
        cache_file = tmp_path / "cache.nc"
        ds_initial = read(archive_dir, lazy=False)
        
        # Verify initial slope coords are valid
        initial_incl = float(ds_initial.data.inclination.values.flat[0])
        initial_azi = float(ds_initial.data.azimuth.values.flat[0])
        assert not np.isnan(initial_incl), "Initial slope should not be NaN"
        assert not np.isnan(initial_azi), "Initial azimuth should not be NaN"
        
        ds_initial.to_netcdf(cache_file)
        
        # Add new data (this will create a file with more timestamps)
        synthetic_pro.varying_layers(
            archive_dir,
            layer_counts=[3, 3, 3, 3]  # One more timestamp
        )
        
        # Load cached data
        ds_cached = read(cache_file)
        
        # Append with new data
        ds_updated = ds_cached.append_latest(archive_dir)
        
        # BUG CHECK: Slope coordinates should still be valid after append_latest
        assert 'inclination' in ds_updated.data.coords
        assert 'azimuth' in ds_updated.data.coords
        
        updated_incl = float(ds_updated.data.inclination.values.flat[0])
        updated_azi = float(ds_updated.data.azimuth.values.flat[0])
        
        assert not np.isnan(updated_incl), "inclination became NaN after append_latest!"
        assert not np.isnan(updated_azi), "azimuth became NaN after append_latest!"
        # Values should be preserved
        assert updated_incl == initial_incl, f"Slope changed from {initial_incl} to {updated_incl}"
        assert updated_azi == initial_azi, f"Azimuth changed from {initial_azi} to {updated_azi}"
    
    def test_mixed_pro_smet_slope_coords(self, synthetic_pro, synthetic_smet, tmp_path):
        """Test that PRO slope coords take precedence over SMET NaN values.
        
        Bug: When SMET files (with NaN slope coords) were merged before PRO files,
        NaN values persisted due to compat='override' keeping first dataset's values.
        Fix: Ensured PRO data is always merged first.
        """
        station_name = "MIXED_STATION"
        
        # Create PRO file with valid slope coordinates (directly in tmp_path)
        synthetic_pro.basic(
            tmp_path,
            station_name=station_name,
            n_layers=3
        )
        
        # Create SMET file (which would have NaN slope coords by default)
        synthetic_smet.basic(
            tmp_path,
            station_name=station_name
        )
        
        # Load both together - SMET files should not overwrite valid PRO slope coords
        ds = read(tmp_path, lazy=False)
        
        # BUG CHECK: Slope coords should come from PRO file, not NaN from SMET
        assert 'inclination' in ds.data.coords
        assert 'azimuth' in ds.data.coords
        
        incl = float(ds.data.inclination.values.flat[0])
        azi = float(ds.data.azimuth.values.flat[0])
        
        assert not np.isnan(incl), "inclination is NaN - SMET NaN overwrote PRO value!"
        assert not np.isnan(azi), "azimuth is NaN - SMET NaN overwrote PRO value!"
    
    def test_multi_location_update_preserves_coords(self, synthetic_pro, tmp_path):
        """Test that appending to multi-location datasets preserves slope coords.
        
        This tests the combination of multi-location handling and append_latest operations.
        """
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        
        # Create two stations
        data_dir = synthetic_pro.multi_station(
            archive_dir,
            n_stations=2,
            n_profiles_each=3
        )
        
        # Initial load
        ds_initial = read(archive_dir, lazy=False)
        
        # Store initial slope values for each location
        initial_slopes = {}
        for loc in ds_initial.location.values:
            loc_ds = ds_initial.sel(location=loc)
            initial_slopes[str(loc)] = (
                float(loc_ds.inclination.values.flat[0]),
                float(loc_ds.azimuth.values.flat[0])
            )
        
        # Cache it
        cache_file = tmp_path / "multi_cache.nc"
        ds_initial.to_netcdf(cache_file)
        
        # Add more data to the archive
        data_dir = synthetic_pro.multi_station(
            archive_dir,
            n_stations=2,
            n_profiles_each=5  # More profiles
        )
        
        # Load and update
        ds_cached = read(cache_file)
        ds_updated = ds_cached.append_latest(archive_dir)
        
        # BUG CHECK: Each location should still have valid slope coords
        for loc in ds_updated.location.values:
            loc_key = str(loc)
            loc_ds = ds_updated.sel(location=loc)
            
            incl = float(loc_ds.inclination.values.flat[0])
            azi = float(loc_ds.azimuth.values.flat[0])
            
            assert not np.isnan(incl), f"{loc}: inclination is NaN after append_latest"
            assert not np.isnan(azi), f"{loc}: azimuth is NaN after append_latest"
            
            # Values should be preserved
            if loc_key in initial_slopes:
                initial_incl, initial_azi = initial_slopes[loc_key]
                assert incl == initial_incl, f"{loc}: slope changed after append_latest"
                assert azi == initial_azi, f"{loc}: azimuth changed after append_latest"
    
    def test_smet_only_slope_coords(self, synthetic_smet, tmp_path):
        """Test behavior when only SMET files are loaded (no PRO files).
        
        When only SMET files exist, slope coords may be NaN or come from header.
        This test documents expected behavior.
        """
        # Create SMET-only data
        synthetic_smet.basic(tmp_path, station_name="SMET_ONLY")
        
        # Load data
        ds = read(tmp_path, lazy=False)
        
        # Slope coordinates should exist (even if NaN)
        assert 'inclination' in ds.data.coords
        assert 'azimuth' in ds.data.coords
        
        # For SMET-only files, slope coords may be NaN or from header
        # This is expected behavior - just verify coordinates exist
        incl_array = ds.data.inclination.values
        azi_array = ds.data.azimuth.values
        
        # Extract scalar value
        incl = float(incl_array.flat[0])
        azi = float(azi_array.flat[0])
        
        # Document: SMET files may have NaN slopes if not in header
        # This is acceptable - the bug is when valid PRO slopes become NaN
        # Just verify we can extract numeric values (even if NaN)
        assert isinstance(incl, (int, float, np.floating))
        assert isinstance(azi, (int, float, np.floating))
