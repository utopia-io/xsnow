# tests/integration/test_real_data.py
"""
Tests using real SNOWPACK model output.
Focus: Real-world data compatibility.
"""
import pytest
import numpy as np
import pandas as pd
from xsnow import read
from xsnow.selectors import TimeSelector, LocationSelector, SlopeSelector


@pytest.mark.real_data
class TestReadRealSnowpackData:
    """Tests for reading real SNOWPACK model output."""
    
    def test_read_snp_gridded_data(self, gridded_pros_dir, assert_physical_validity):
        """Test reading gridded SNOWPACK .pro files."""
        ds = read(gridded_pros_dir, lazy=False)
        
        assert ds is not None
        assert ds.sizes['location'] > 0
        assert ds.sizes['time'] > 0
        
        # Check required variables
        assert 'height' in ds.data
        assert 'density' in ds.data
        assert 'temperature' in ds.data
        
        # Validate physical constraints
        assert_physical_validity(ds)
    
    def test_read_snp_snowobs_data(self, snowobs_pros_dir, assert_physical_validity):
        """Test reading snow observation SNOWPACK files."""
        ds = read(snowobs_pros_dir, lazy=False)
        
        assert ds is not None
        assert ds.sizes['location'] > 0
        
        # Should have metadata
        assert 'latitude' in ds.data.coords
        assert 'longitude' in ds.data.coords
        assert 'altitude' in ds.data.coords
        
        assert_physical_validity(ds)
    
    def test_read_single_real_pro(self, sample_pro_file, assert_physical_validity):
        """Test reading a single real .pro file."""
        ds = read(sample_pro_file)
        
        assert ds is not None
        assert ds.sizes['location'] == 1
        assert ds.sizes['time'] > 0
        
        # Check that we have layer data
        assert 'layer' in ds.data.dims
        assert ds.sizes['layer'] > 0
        
        assert_physical_validity(ds)
    
    def test_read_real_smet(self, snowobs_smets_dir_nowcast):
        """Test reading real SMET meteorological files."""
        ds = read(snowobs_smets_dir_nowcast)
        
        if ds is None:
            pytest.skip("No SMET files found in test data")
        
        assert 'time' in ds.data.dims
        # Should have some meteorological variables
        assert len(ds.data.data_vars) > 0


@pytest.mark.real_data
class TestRealDataPhysicalValidity:
    """Validate physical realism of real SNOWPACK data."""
    
    def test_real_data_physical_validity(self, sample_pro_file):
        """Test that real data passes physical validity checks."""
        ds = read(sample_pro_file)
        
        # Density should be realistic
        if 'density' in ds.data:
            density = ds['density'].values
            valid_density = density[~np.isnan(density)]
            # Fresh snow can be as low as 6-7 kg/m³
            assert np.all(valid_density >= 5), "Unrealistic low density"
            assert np.all(valid_density <= 917), "Density exceeds ice density"
        
        # Temperature should be reasonable
        if 'temperature' in ds.data:
            temp = ds['temperature'].values
            valid_temp = temp[~np.isnan(temp)]
            if valid_temp.max() > 100:  # Kelvin
                assert np.all(valid_temp >= 200), "Temperature too low"
                assert np.all(valid_temp <= 280), "Temperature too high for snow"
            else:  # Celsius
                assert np.all(valid_temp >= -50), "Temperature too low"
                assert np.all(valid_temp <= 5), "Temperature too high for snow"
        
        # Heights should be non-negative and reasonable
        if 'height' in ds.data:
            height = ds['height'].values
            valid_height = height[~np.isnan(height)]
            assert np.all(valid_height >= 0), "Negative heights found"
            assert np.all(valid_height <= 20), "Unrealistic snow depth"
        
        # Grain sizes should be reasonable
        if 'grain_size' in ds.data:
            grain_size = ds['grain_size'].values
            valid_gs = grain_size[~np.isnan(grain_size)]
            assert np.all(valid_gs >= 0), "Negative grain size"
            assert np.all(valid_gs <= 50), "Unrealistic grain size"
    
    def test_real_data_temporal_continuity(self, sample_pro_file):
        """Test temporal continuity of real data."""
        ds = read(sample_pro_file)
        
        # Timestamps should be monotonically increasing
        times = pd.to_datetime(ds.data.time.values)
        assert times.is_monotonic_increasing, "Timestamps not ordered"
        
        # Check reasonable time spacing
        if len(times) > 1:
            diffs = np.diff(times).astype('timedelta64[h]').astype(int)
            assert np.all(diffs > 0), "Zero or negative time steps"
            assert np.all(diffs <= 24 * 365), "Unrealistic time gaps (>1 year)"
    
    def test_real_data_vertical_structure(self, sample_pro_file):
        """Test vertical structure makes physical sense."""
        ds = read(sample_pro_file)
        
        if 'height' not in ds.data or 'density' not in ds.data:
            pytest.skip("Required variables not present")
        
        # Check one profile
        height = ds['height'].isel(location=0, time=0, slope=0, realization=0).values
        density = ds['density'].isel(location=0, time=0, slope=0, realization=0).values
        
        # Remove NaN values
        valid = ~(np.isnan(height) | np.isnan(density))
        height = height[valid]
        density = density[valid]
        
        if len(height) > 1:
            # Heights should generally increase (though may have inversions)
            assert height[-1] > height[0], "Height profile inverted"
            
            # Density should generally increase with depth (lower heights)
            # but this is not strict, so we just check it's variable
            assert density.std() > 0, "No density variation"


@pytest.mark.real_data
class TestRealDataMultiSlope:
    """Tests for real multi-slope datasets."""
    
    def test_real_data_multi_slope(self, gridded_pros_dir):
        """Test that multi-slope data is correctly loaded and merged."""
        ds = read(gridded_pros_dir, lazy=False)
        
        # Gridded data should have multiple slopes
        if 'slope' in ds.data.dims:
            assert ds.sizes['slope'] > 1, "Expected multiple slopes in gridded data"
            
            # Check slope metadata
            assert 'inclination' in ds.data.coords
            slope_angles = ds.data['inclination'].values
            
            # Should have different slope angles
            unique_slopes = np.unique(slope_angles[~np.isnan(slope_angles)])
            assert len(unique_slopes) > 1, "Expected different slope angles"
    
    def test_real_data_slope_consistency(self, gridded_pros_dir):
        """Test that data across slopes is handled correctly."""
        ds = read(gridded_pros_dir, lazy=False)
        
        if 'slope' not in ds.data.dims or ds.sizes['slope'] < 2:
            pytest.skip("Not enough slopes for comparison")
        
        # In gridded data, each "slope" may actually represent a different location
        # Check that we have valid metadata for all slopes
        if 'altitude' in ds.data.coords:
            altitude_by_slope = ds.data['altitude'].values
            # All altitudes should be valid (not NaN)
            assert np.all(~np.isnan(altitude_by_slope)), "Found NaN altitudes"
            # Altitudes should be reasonable
            assert np.all(altitude_by_slope > 0), "Invalid altitude values"
            assert np.all(altitude_by_slope < 9000), "Unrealistic altitude values"


@pytest.mark.real_data
class TestRealDataWithFilters:
    """Test filtering operations on real data."""
    
    def test_real_data_with_filters(self, sample_pro_file):
        """Test various selectors on real data."""
        # Load full dataset
        ds_full = read(sample_pro_file)
        
        if ds_full.sizes['time'] < 2:
            pytest.skip("Not enough timesteps for filtering")
        
        # Get time range
        times = pd.to_datetime(ds_full.data.time.values)
        mid_time = times[len(times) // 2]
        
        # Apply time filter
        time_selector = TimeSelector(sel=slice(None, mid_time.isoformat()))
        ds_filtered = read(sample_pro_file, time=time_selector)
        
        assert ds_filtered.sizes['time'] <= ds_full.sizes['time']
        assert ds_filtered.sizes['time'] > 0
    
    def test_real_data_location_filter(self, gridded_pros_dir):
        """Test location filtering on real data."""
        ds_full = read(gridded_pros_dir, lazy=False)
        
        if ds_full.sizes['location'] < 2:
            pytest.skip("Only one location in dataset")
        
        # Get first location name
        location_map = ds_full.location.attrs.get('mapping', {})
        if not location_map:
            pytest.skip("No location mapping available")
        
        first_location = location_map[0]
        
        # Filter to first location
        loc_selector = LocationSelector(sel=first_location)
        ds_filtered = read(gridded_pros_dir, location=loc_selector, lazy=False)
        
        # Location dimension may be squeezed when selecting single location
        if 'location' in ds_filtered.sizes:
            assert ds_filtered.sizes['location'] == 1
        else:
            # Dimension was squeezed - verify we still have data
            assert ds_filtered.data is not None
            assert len(ds_filtered.data.data_vars) > 0
    
    def test_real_data_slope_filter(self, gridded_pros_dir):
        """Test slope filtering on real multi-slope data."""
        ds_full = read(gridded_pros_dir, lazy=False)
        
        if 'slope' not in ds_full.data.dims or ds_full.sizes['slope'] < 2:
            pytest.skip("Not enough slopes for filtering")
        
        # Filter to first slope
        slope_selector = SlopeSelector(sel=0)
        ds_filtered = read(gridded_pros_dir, slope=slope_selector, lazy=False)
        
        # Slope dimension may be squeezed when selecting single slope
        if 'slope' in ds_filtered.sizes:
            assert ds_filtered.sizes['slope'] == 1
        else:
            # Dimension was squeezed - verify we still have data
            assert ds_filtered.data is not None
            assert len(ds_filtered.data.data_vars) > 0

@pytest.mark.real_data
class TestRealDataConsistency:
    """Test consistency between loading modes."""
    
    def test_real_data_consistency_eager_vs_lazy(self, real_file_subset, compare_datasets):
        """Test that eager and lazy loading produce identical results."""
        # Load with eager mode
        ds_eager = read(real_file_subset, lazy=False)
        
        # Load with lazy mode
        ds_lazy = read(real_file_subset, lazy=True)
        
        # Should produce same dimensions
        assert ds_eager.sizes == ds_lazy.sizes
        
        # Should produce same data values
        compare_datasets(ds_eager, ds_lazy, rtol=1e-6)
    
    def test_real_data_reproducibility(self, sample_pro_file):
        """Test that repeated loads produce identical results."""
        ds1 = read(sample_pro_file, lazy=False)
        ds2 = read(sample_pro_file, lazy=False)
        
        # Should be exactly equal
        assert ds1.sizes == ds2.sizes
        
        for var in ds1.data.data_vars:
            if var in ds2.data.data_vars:
                np.testing.assert_array_equal(
                    ds1[var].values,
                    ds2[var].values,
                    err_msg=f"Variable {var} differs between loads"
                )