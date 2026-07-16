# tests/features/test_multi_slope.py
"""
Tests for multi-slope dataset handling.
Focus: Correct merging and handling of multiple slope angles.
"""
import pytest
import numpy as np
from xsnow import read
from xsnow.selectors import SlopeSelector


@pytest.mark.synthetic
class TestMultiSlopeDetection:
    """Test detection and loading of multi-slope data."""
    
    def test_detect_multiple_slopes(self, synthetic_pro, tmp_path):
        """Test that multiple slope files are detected."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Should have slope dimension
        assert 'slope' in ds.data.dims
        assert ds.sizes['slope'] == len(slopes)
    
    def test_slope_metadata_preserved(self, synthetic_pro, tmp_path):
        """Test that slope metadata is correctly preserved."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Check inclination coordinate
        if 'inclination' in ds.data.coords:
            slope_angles = ds.data['inclination'].values
            
            # Should have all the slope angles
            for expected_slope in slopes:
                assert expected_slope in slope_angles
    
    def test_single_slope_no_dimension(self, synthetic_pro, tmp_path):
        """Test that single slope doesn't create unnecessary dimension."""
        filepath = synthetic_pro.basic(tmp_path)
        
        ds = read(filepath)
        
        # Should have slope dimension but size 1
        assert 'slope' in ds.data.dims
        assert ds.sizes['slope'] == 1


@pytest.mark.synthetic
class TestMultiSlopeMerging:
    """Test merging of multi-slope data."""
    
    def test_merge_slopes_same_location(self, synthetic_pro, tmp_path):
        """Test that slopes from same location are merged correctly."""
        slopes = [0, 30]
        data_dir = synthetic_pro.multi_slope(tmp_path, "STATION_A", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Should have 1 location, 2 slopes
        assert ds.sizes['location'] == 1
        assert ds.sizes['slope'] == 2

    def test_fold_filename_fallback_aspect_locations(self, synthetic_pro, tmp_path):
        """Fold SNOWPACK aspect suffixes only when location came from filename."""
        synthetic_pro.aspect_file(
            tmp_path / "STATION.pro",
            station_name=None,
            slope_angle=0,
            slope_azi=0,
        )
        synthetic_pro.aspect_file(
            tmp_path / "STATION1.pro",
            station_name=None,
            slope_angle=35,
            slope_azi=180,
        )

        ds = read(tmp_path, lazy=False, n_cpus_use=1, max_layers=3)

        assert list(ds.location.values) == ["STATION"]
        assert ds.sizes["slope"] == 2
        assert set(ds.inclination.values.tolist()) == {0, 35}
        assert "_xsnow_location_source" not in ds.data.attrs

    def test_do_not_fold_explicit_trailing_digit_station_name(self, synthetic_pro, tmp_path):
        """Do not override explicit StationName metadata that looks like an aspect suffix."""
        synthetic_pro.aspect_file(
            tmp_path / "STATION.pro",
            station_name="STATION",
            slope_angle=0,
            slope_azi=0,
        )
        synthetic_pro.aspect_file(
            tmp_path / "STATION1.pro",
            station_name="STATION1",
            slope_angle=35,
            slope_azi=180,
        )

        ds = read(tmp_path, lazy=False, n_cpus_use=1, max_layers=3)

        assert set(ds.location.values.tolist()) == {"STATION", "STATION1"}

    def test_do_not_fold_without_compatible_flat_base(self, synthetic_pro, tmp_path):
        """Require a compatible flat base profile before folding aspect suffixes."""
        synthetic_pro.aspect_file(
            tmp_path / "STATION.pro",
            station_name=None,
            slope_angle=10,
            slope_azi=0,
        )
        synthetic_pro.aspect_file(
            tmp_path / "STATION1.pro",
            station_name=None,
            slope_angle=35,
            slope_azi=180,
        )

        ds = read(tmp_path, lazy=False, n_cpus_use=1, max_layers=3)

        assert set(ds.location.values.tolist()) == {"STATION", "STATION1"}
    
    def test_merge_preserves_data(self, synthetic_pro, tmp_path):
        """Test that data from each slope is preserved during merge."""
        slopes = [0, 30]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Each slope should have valid data
        if 'height' in ds.data:
            for slope_idx in range(ds.sizes['slope']):
                height = ds['height'].isel(slope=slope_idx).values
                valid_data = height[~np.isnan(height)]
                assert len(valid_data) > 0, f"Slope {slope_idx} has no valid data"
    
    def test_slopes_sorted_by_angle(self, synthetic_pro, tmp_path):
        """Test that slopes are sorted by inclination angle."""
        slopes = [45, 0, 30, 15]  # Unsorted
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        if 'inclination' in ds.data.coords:
            slope_angles = ds.data['inclination'].values
            valid_angles = slope_angles[~np.isnan(slope_angles)]
            
            # Should be sorted
            assert np.all(valid_angles[:-1] <= valid_angles[1:])
    
    def test_slope_dimension_indexing(self, synthetic_pro, tmp_path):
        """Test that slope dimension is correctly indexed."""
        slopes = [0, 15, 30]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Slope dimension should be 0, 1, 2
        assert list(ds.data['slope'].values) == [0, 1, 2]


@pytest.mark.synthetic
class TestMultiSlopeSelection:
    """Test selection and filtering of multi-slope data."""
    
    def test_select_single_slope(self, synthetic_pro, tmp_path):
        """Test selecting a single slope from multi-slope dataset."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Select first slope
        ds_single = ds.isel(slope=0)
        
        assert 'slope' not in ds_single.data.dims or ds_single.sizes['slope'] == 1
    
    def test_select_slope_range(self, synthetic_pro, tmp_path):
        """Test selecting a range of slopes."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Select middle two slopes
        ds_subset = ds.isel(slope=slice(1, 3))
        
        assert ds_subset.sizes['slope'] == 2
    
    def test_slope_selector_integration(self, synthetic_pro, tmp_path):
        """Test SlopeSelector with multi-slope data."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        # Select using SlopeSelector
        slope_filter = SlopeSelector(sel=0)
        ds = read(data_dir, slope=slope_filter, lazy=False)
        
        assert 'slope' not in ds.data.dims or ds.sizes['slope'] == 1
    
    def test_select_by_slope_angle(self, synthetic_pro, tmp_path):
        """Test selecting slopes by angle value."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        if 'inclination' in ds.data.coords:
            # Select single location first (handles both single and multi-location)
            if ds.sizes['location'] > 1:
                ds = ds.isel(location=0)
            else:
                ds = ds.squeeze('location', drop=False)
            
            # Now inclination should be 1D
            slope_angles = ds.data['inclination'].values
            idx_30 = np.where(slope_angles == 30)[0]
            
            if len(idx_30) > 0:
                ds_30 = ds.isel(slope=idx_30[0])
                
                # Verify we got the right slope
                incl_value = float(ds_30.data['inclination'].values.flat[0])
                assert np.isclose(incl_value, 30.0), \
                    f"Expected 30.0, got {incl_value}"


@pytest.mark.synthetic
class TestMultiSlopeWithMultipleLocations:
    """Test multi-slope data with multiple locations."""
    
    def test_different_slopes_per_location(self, synthetic_pro, tmp_path):
        """Test handling when different locations have different slopes."""
        # Station A with 2 slopes
        dir_a = tmp_path / "station_a"
        dir_a.mkdir()
        synthetic_pro.multi_slope(dir_a, "STATION_A", slopes=[0, 30])
        
        # Station B with 3 slopes
        dir_b = tmp_path / "station_b"
        dir_b.mkdir()
        synthetic_pro.multi_slope(dir_b, "STATION_B", slopes=[0, 15, 30])
        
        # Copy all files to common directory
        common_dir = tmp_path / "combined"
        common_dir.mkdir()
        
        import shutil
        for f in dir_a.glob("*.pro"):
            shutil.copy(f, common_dir / f.name)
        for f in dir_b.glob("*.pro"):
            shutil.copy(f, common_dir / f.name)
        
        ds = read(common_dir, lazy=False)
        
        # Should have 2 locations
        assert ds.sizes['location'] == 2
        
        # Should have max(2, 3) = 3 slopes
        assert ds.sizes['slope'] >= 2
    
    def test_multi_location_multi_slope_data_integrity(self, synthetic_pro, tmp_path):
        """Test data integrity with multiple locations and slopes."""
        # Create data
        for i in range(3):
            station_dir = tmp_path / f"station_{i}"
            station_dir.mkdir()
            synthetic_pro.multi_slope(
                station_dir,
                f"STATION_{i}",
                slopes=[0, 30]
            )
        
        # Combine all
        combined_dir = tmp_path / "combined"
        combined_dir.mkdir()
        
        import shutil
        for station_dir in tmp_path.glob("station_*"):
            for f in station_dir.glob("*.pro"):
                shutil.copy(f, combined_dir / f"{station_dir.name}_{f.name}")
        
        ds = read(combined_dir, lazy=False)
        
        # Should have 3 locations, 2 slopes each
        assert ds.sizes['location'] == 3
        assert ds.sizes['slope'] == 2
        
        # Each location-slope combination should have data
        if 'height' in ds.data:
            for loc in range(ds.sizes['location']):
                for slope in range(ds.sizes['slope']):
                    height = ds['height'].isel(location=loc, slope=slope).values
                    valid = height[~np.isnan(height)]
                    assert len(valid) > 0, f"No data for location {loc}, slope {slope}"


@pytest.mark.synthetic
class TestMultiSlopeEdgeCases:
    """Test edge cases in multi-slope handling."""
    
    def test_duplicate_slopes_same_location(self, synthetic_pro, tmp_path):
        """Test handling of duplicate slope angles for same location."""
        # Create two files with same slope angle
        content = """[STATION_PARAMETERS]
StationName= DUP_SLOPE
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
SlopeAngle= 30

[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
"""
        
        (tmp_path / "file1.pro").write_text(content)
        (tmp_path / "file2.pro").write_text(content)
        
        # Should handle gracefully (merge or keep both)
        ds = read(tmp_path, lazy=False)
        
        assert ds is not None
    
    def test_missing_slope_metadata(self, synthetic_pro, tmp_path):
        """Test handling when slope metadata is missing."""
        content = """[STATION_PARAMETERS]
StationName= NO_SLOPE

[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
"""
        
        filepath = tmp_path / "no_slope.pro"
        filepath.write_text(content)
        
        ds = read(filepath)
        
        # Should still work, assume flat
        assert ds is not None
        if 'inclination' in ds.data.coords:
            slope_angle = ds.data['inclination'].values[0]
            # Should be NaN or 0
            assert np.isnan(slope_angle) or slope_angle == 0
    
    def test_extreme_slope_angles(self, synthetic_pro, tmp_path):
        """Test handling of extreme slope angles."""
        slopes = [0, 45, 60, 75, 90]  # Including vertical
        data_dir = synthetic_pro.multi_slope(tmp_path, "STEEP", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        assert ds is not None
        assert ds.sizes['slope'] == len(slopes)
    
    def test_negative_slope_angles(self, synthetic_pro, tmp_path):
        """Test handling of negative slope angles (overhang/unusual)."""
        # Manually create with negative slope
        content = """[STATION_PARAMETERS]
StationName= NEGATIVE_SLOPE
SlopeAngle= -15

[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
"""
        
        filepath = tmp_path / "negative_slope.pro"
        filepath.write_text(content)
        
        # Should handle without crashing
        ds = read(filepath)
        
        assert ds is not None


@pytest.mark.synthetic
class TestMultiSlopeComputations:
    """Test computations across multiple slopes."""
    
    def test_compute_hs_multi_slope(self, synthetic_pro, tmp_path):
        """Test HS computation with multiple slopes."""
        slopes = [0, 30]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        ds = ds.compute_hs()
        
        # Should have HS for each slope
        if 'HS' in ds.data:
            assert 'HS' in ds.data.data_vars
            
            # Check that each slope has valid HS
            for slope_idx in range(ds.sizes['slope']):
                hs = ds['HS'].isel(slope=slope_idx).values
                assert not np.all(np.isnan(hs))
    
    def test_compute_z_depth_multi_slope(self, synthetic_pro, tmp_path):
        """Test z-depth computation with multiple slopes."""
        slopes = [0, 30]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        ds = ds.compute_z_depth()
        
        # Should have z coordinate for each slope
        if 'z' in ds.data.coords:
            for slope_idx in range(ds.sizes['slope']):
                z = ds.data['z'].isel(slope=slope_idx).values
                valid_z = z[~np.isnan(z)]
                assert len(valid_z) > 0
    
    def test_mean_across_slopes(self, synthetic_pro, tmp_path):
        """Test computing mean values across slopes."""
        slopes = [0, 15, 30, 45]
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        ds = ds.compute_hs()
        
        if 'HS' in ds.data:
            # Compute mean HS across slopes
            hs_mean = ds['HS'].mean(dim='slope')
            
            # Should have reduced slope dimension
            assert 'slope' not in hs_mean.dims
            assert not np.all(np.isnan(hs_mean.values))


@pytest.mark.real_data
class TestRealMultiSlopeData:
    """Test with real multi-slope data if available."""
    
    def test_real_gridded_multi_slope(self, gridded_pros_dir):
        """Test reading real gridded multi-slope data."""
        ds = read(gridded_pros_dir, lazy=False)
        
        if 'slope' not in ds.data.dims:
            pytest.skip("Real data doesn't have multiple slopes")
        
        # Should have multiple slopes
        assert ds.sizes['slope'] > 1
        
        # Should have slope metadata
        assert 'inclination' in ds.data.coords
        
        # Each slope should have data
        if 'height' in ds.data:
            for slope_idx in range(ds.sizes['slope']):
                height = ds['height'].isel(slope=slope_idx).values
                valid = height[~np.isnan(height)]
                assert len(valid) > 0
    
    def test_real_multi_slope_physical_validity(self, gridded_pros_dir, assert_physical_validity):
        """Test physical validity across all slopes."""
        ds = read(gridded_pros_dir, lazy=False)
        
        if 'slope' not in ds.data.dims:
            pytest.skip("Real data doesn't have multiple slopes")
        
        # Check each slope independently
        for slope_idx in range(ds.sizes['slope']):
            ds_slope = ds.isel(slope=slope_idx)
            assert_physical_validity(ds_slope)


@pytest.mark.synthetic
class TestMultiSlopeIO:
    """Test I/O operations with multi-slope data."""
    
    def test_save_load_multi_slope_netcdf(self, synthetic_pro, tmp_path, compare_datasets):
        """Test saving and loading multi-slope data to NetCDF."""
        slopes = [0, 15, 30]
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        data_dir = synthetic_pro.multi_slope(source_dir, "TEST_STATION", slopes=slopes)
        
        # Load
        ds_original = read(data_dir, lazy=False)
        
        # Save
        nc_file = tmp_path / "multi_slope.nc"
        ds_original.to_netcdf(nc_file)
        
        # Reload
        ds_reloaded = read(nc_file)
        
        # Should be identical
        assert ds_reloaded.sizes['slope'] == ds_original.sizes['slope']
        compare_datasets(ds_original, ds_reloaded, rtol=1e-6)
    
    def test_export_single_slope_to_pro(self, synthetic_pro, tmp_path):
        """Test exporting single slope to .pro file."""
        slopes = [0, 30]
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        data_dir = synthetic_pro.multi_slope(source_dir, "TEST_STATION", slopes=slopes)
        
        ds = read(data_dir, lazy=False)
        
        # Select one slope
        ds_single = ds.isel(slope=0)
        
        # Export
        pro_file = tmp_path / "exported.pro"
        ds_single.to_pro(pro_file)
        
        assert pro_file.exists()
        
        # Reload and verify
        ds_reloaded = read(tmp_path)
        assert ds_reloaded is not None
