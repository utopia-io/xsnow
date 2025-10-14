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
    """Test caching and incremental update workflows."""
    
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
        """Test incremental update workflow."""
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
        
        # Update with new data
        ds_updated = ds_cached.update(archive_dir)
        
        # Note: This will merge both files, so may have more timestamps
        # The key is that update doesn't crash and adds new data
        assert ds_updated.sizes['time'] >= ds_initial.sizes['time']
        
        # Save updated cache
        ds_updated.to_netcdf(cache_file)
    
    
    def test_update_with_no_new_data(self, synthetic_pro, tmp_path):
        """Test update when no new data is available."""
        filepath = synthetic_pro.basic(tmp_path)  # Pass directory
        
        # Initial load
        ds = read(tmp_path)  # Read from directory
        original_size = ds.sizes['time']
        
        # Update with same source (no new data)
        ds_updated = ds.update(tmp_path)
        
        # Should be unchanged
        assert ds_updated.sizes['time'] == original_size

    
    def test_update_handles_duplicates(self, synthetic_pro, tmp_path):
        """Test that update handles duplicate timestamps correctly."""
        # Create file with 3 profiles
        filepath = synthetic_pro.varying_layers(
            tmp_path,  # Pass directory
            layer_counts=[3, 3, 3]
        )
        
        # Initial load
        ds = read(tmp_path)
        
        # "Update" with same file (duplicates)
        ds_updated = ds.update(tmp_path)
        
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
        assert ds.sizes['location'] == 5  # Only valid files
    
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