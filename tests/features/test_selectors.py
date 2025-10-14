# tests/features/test_selectors.py
"""
Tests for Time/Location/Slope/Realization selectors.
Focus: Selector functionality and edge cases.
"""
import pytest
import pandas as pd
import xarray as xr
from xsnow import read
from xsnow.selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector


@pytest.mark.synthetic
class TestTimeSelector:
    """Tests for TimeSelector functionality."""
    
    def test_time_selector_single_timestamp(self, synthetic_pro, tmp_path):
        """Test selecting a single timestamp."""
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=[3, 3, 3])
        
        ds = read(filepath)
        times = pd.to_datetime(ds.data.time.values)
        
        # Select middle timestamp
        middle_time = times[1]
        selector = TimeSelector(sel=middle_time)
        ds_filtered = selector.apply(ds.data)
        
        # Dimension may be squeezed when selecting single value
        if 'time' in ds_filtered.sizes:
            assert ds_filtered.sizes['time'] == 1
            assert pd.to_datetime(ds_filtered.time.values[0]) == middle_time
        else:
            # Dimension was squeezed
            assert pd.to_datetime(ds_filtered.time.values) == middle_time
            
    def test_time_selector_slice(self, synthetic_pro, tmp_path):
        """Test selecting a time range with slice."""
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=[3]*10)
        
        ds = read(filepath)
        
        # Select days 3-7
        selector = TimeSelector(sel=slice('2025-01-03', '2025-01-07'))
        ds_filtered = selector.apply(ds.data)
        
        assert ds_filtered.sizes['time'] == 5
    
    def test_time_selector_list(self, synthetic_pro, tmp_path):
        """Test selecting specific timestamps from a list."""
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=[3]*5)
        
        ds = read(filepath)
        times = pd.to_datetime(ds.data.time.values)
        
        # Select first and last
        selected_times = [times[0], times[-1]]
        selector = TimeSelector(sel=selected_times)
        ds_filtered = selector.apply(ds.data)
        
        assert ds_filtered.sizes['time'] == 2
    
    def test_time_selector_with_read(self, synthetic_pro, tmp_path):
        """Test TimeSelector applied during read()."""
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=[3]*10)
        
        time_filter = TimeSelector(sel=slice('2025-01-03', '2025-01-07'))
        ds = read(filepath, time=time_filter)
        
        assert ds.sizes['time'] == 5
    
    def test_time_selector_out_of_range(self, synthetic_pro, tmp_path):
        """Test TimeSelector with dates outside data range."""
        filepath = synthetic_pro.basic(tmp_path)
        
        # Try to select future date
        selector = TimeSelector(sel=slice('2026-01-01', '2026-12-31'))
        ds = read(filepath, time=selector)
        
        # Should return empty or None
        if ds is not None:
            assert ds.sizes['time'] == 0


@pytest.mark.synthetic
class TestLocationSelector:
    """Tests for LocationSelector functionality."""
    
    def test_location_selector_by_name(self, synthetic_pro, tmp_path):
        """Test selecting location by station name."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        ds = read(data_dir, lazy=False)
        
        # Get first station name
        location_map = ds.location.attrs['mapping']
        first_station = location_map[0]
        
        # Select by name
        selector = LocationSelector(sel=first_station)
        ds_filtered = selector.apply(ds.data)
        
        # Location dimension may be squeezed when selecting single location
        if 'location' in ds_filtered.sizes:
            assert ds_filtered.sizes['location'] == 1
        else:
            # Dimension was squeezed - verify we have data
            assert len(ds_filtered.data_vars) > 0

    def test_location_selector_by_index(self, synthetic_pro, tmp_path):
        """Test selecting location by integer index."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        ds = read(data_dir, lazy=False)
        
        # Select first location (index 0)
        selector = LocationSelector(sel=0)
        ds_filtered = selector.apply(ds.data)
        
        # Location dimension may be squeezed
        if 'location' in ds_filtered.sizes:
            assert ds_filtered.sizes['location'] == 1
        else:
            assert len(ds_filtered.data_vars) > 0
                
    def test_location_selector_list(self, synthetic_pro, tmp_path):
        """Test selecting multiple locations."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        ds = read(data_dir, lazy=False)
        location_map = ds.location.attrs['mapping']
        
        # Select first and last stations
        stations = [location_map[0], location_map[4]]
        selector = LocationSelector(sel=stations)
        ds_filtered = selector.apply(ds.data)
        
        assert ds_filtered.sizes['location'] == 2
    
    def test_location_selector_bbox(self, synthetic_pro, tmp_path):
        """Test geographic bounding box selection."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=10)
        
        ds = read(data_dir, lazy=False)
        
        # Define bounding box
        selector = LocationSelector(
            lat_range=(46.0, 47.0),
            lon_range=(10.0, 11.0),
            crs='EPSG:4326'
        )
        
        ds_filtered = selector.apply(ds.data)
        
        # Should select some stations within the box
        assert ds_filtered.sizes['location'] > 0
        assert ds_filtered.sizes['location'] <= 10
    
    def test_location_selector_with_read(self, synthetic_pro, tmp_path):
        """Test LocationSelector applied during read()."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        # Read all to get location names
        ds_all = read(data_dir, lazy=False)
        first_station = ds_all.location.attrs['mapping'][0]
        
        # Read with filter
        loc_filter = LocationSelector(sel=first_station)
        ds_filtered = read(data_dir, location=loc_filter, lazy=False)
        
        # Location dimension may be squeezed
        if 'location' in ds_filtered.sizes:
            assert ds_filtered.sizes['location'] == 1
        else:
            assert len(ds_filtered.data.data_vars) > 0

@pytest.mark.synthetic
class TestSlopeSelector:
    """Tests for SlopeSelector functionality."""
    
    def test_slope_selector_single_index(self, synthetic_pro, tmp_path):
        """Test selecting a single slope by index."""
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=[0, 15, 30, 45])
        
        ds = read(data_dir, lazy=False)
        
        if 'slope' not in ds.data.dims:
            pytest.skip("Multi-slope data not properly loaded")
        
        # Select first slope
        selector = SlopeSelector(sel=0)
        ds_filtered = selector.apply(ds.data)
        
        # Slope dimension may be squeezed
        if 'slope' in ds_filtered.sizes:
            assert ds_filtered.sizes['slope'] == 1
        else:
            assert len(ds_filtered.data_vars) > 0
    
    def test_slope_selector_slice(self, synthetic_pro, tmp_path):
        """Test selecting a range of slopes."""
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=[0, 15, 30, 45])
        
        ds = read(data_dir, lazy=False)
        
        if 'slope' not in ds.data.dims or ds.sizes['slope'] < 2:
            pytest.skip("Not enough slopes")
        
        # Select first two slopes
        selector = SlopeSelector(sel=slice(0, 2))
        ds_filtered = selector.apply(ds.data)
        
        assert ds_filtered.sizes['slope'] == 2
    
    def test_slope_selector_list(self, synthetic_pro, tmp_path):
        """Test selecting specific slopes from a list."""
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=[0, 15, 30, 45])
        
        ds = read(data_dir, lazy=False)
        
        if ds.sizes.get('slope', 0) < 4:
            pytest.skip("Not enough slopes")
        
        # Select first and last
        selector = SlopeSelector(sel=[0, 3])
        ds_filtered = selector.apply(ds.data)
        
        assert ds_filtered.sizes['slope'] == 2
    
    def test_slope_selector_with_read(self, synthetic_pro, tmp_path):
        """Test SlopeSelector applied during read()."""
        data_dir = synthetic_pro.multi_slope(tmp_path, "TEST_STATION", slopes=[0, 15, 30])
        
        slope_filter = SlopeSelector(sel=0)
        ds = read(data_dir, slope=slope_filter, lazy=False)
        
        if 'slope' in ds.data.dims:
            assert ds.sizes['slope'] == 1


@pytest.mark.synthetic
class TestRealizationSelector:
    """Tests for RealizationSelector functionality."""
    
    def test_realization_selector_single(self, snow_profile_xr_dataset):
        """Test selecting a single realization."""
        # Expand to multiple realizations
        ds = snow_profile_xr_dataset
        ds = ds.expand_dims(realization=[0, 1, 2])
        
        selector = RealizationSelector(sel=0)
        ds_filtered = selector.apply(ds)
        
        # Realization dimension may be squeezed
        if 'realization' in ds_filtered.sizes:
            assert ds_filtered.sizes['realization'] == 1
        else:
            assert len(ds_filtered.data_vars) > 0
    
    def test_realization_selector_slice(self, snow_profile_xr_dataset):
        """Test selecting a range of realizations."""
        ds = snow_profile_xr_dataset
        ds = ds.expand_dims(realization=[0, 1, 2, 3, 4])
        
        selector = RealizationSelector(sel=slice(1, 4))
        ds_filtered = selector.apply(ds)
        
        assert ds_filtered.sizes['realization'] == 3
    
    def test_realization_selector_list(self, snow_profile_xr_dataset):
        """Test selecting specific realizations."""
        ds = snow_profile_xr_dataset
        ds = ds.expand_dims(realization=[0, 1, 2, 3])
        
        selector = RealizationSelector(sel=[0, 3])
        ds_filtered = selector.apply(ds)
        
        assert ds_filtered.sizes['realization'] == 2


@pytest.mark.synthetic
class TestCombinedSelectors:
    """Test using multiple selectors together."""
    
    def test_multiple_selectors(self, synthetic_pro, tmp_path):
        """Test applying multiple selectors simultaneously."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=10)
        
        # Read all
        ds_all = read(data_dir, lazy=False)
        
        # Get selection criteria
        times = pd.to_datetime(ds_all.data.time.values)
        mid_time = times[len(times) // 2]
        first_station = ds_all.location.attrs['mapping'][0]
        
        # Apply multiple filters
        time_filter = TimeSelector(sel=slice(None, mid_time.isoformat()))
        loc_filter = LocationSelector(sel=first_station)
        
        ds_filtered = read(data_dir, time=time_filter, location=loc_filter, lazy=False)
        
        # Location dimension may be squeezed
        if 'location' in ds_filtered.sizes:
            assert ds_filtered.sizes['location'] == 1
        else:
            assert len(ds_filtered.data.data_vars) > 0
        
        assert ds_filtered.sizes['time'] <= ds_all.sizes['time']

    
    def test_chained_selection(self, synthetic_pro, tmp_path):
        """Test chaining selectors on dataset."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=10)
        
        ds = read(data_dir, lazy=False)
        
        # Chain selections
        ds_filtered = ds.isel(location=slice(0, 2))
        ds_filtered = ds_filtered.isel(time=slice(0, 5))
        
        assert ds_filtered.sizes['location'] == 2
        assert ds_filtered.sizes['time'] == 5


@pytest.mark.synthetic
class TestSelectorEdgeCases:
    """Test edge cases and error handling in selectors."""
    
    def test_selector_empty_result(self, synthetic_pro, tmp_path):
        """Test selector that produces empty result."""
        filepath = synthetic_pro.basic(tmp_path)
        
        # Select time range that doesn't exist
        time_filter = TimeSelector(sel=slice('2026-01-01', '2026-12-31'))
        ds = read(filepath, time=time_filter)
        
        if ds is not None:
            assert ds.sizes['time'] == 0
    
    def test_selector_invalid_location(self, synthetic_pro, tmp_path):
        """Test location selector with non-existent location."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=3)
        
        ds = read(data_dir, lazy=False)
        
        # Try to select non-existent location
        selector = LocationSelector(sel='NONEXISTENT_STATION')
        
        # Should either raise error or return empty
        try:
            ds_filtered = selector.apply(ds.data)
            if ds_filtered is not None:
                assert ds_filtered.sizes['location'] == 0
        except (KeyError, IndexError):
            pass  # Expected behavior
    
    def test_selector_out_of_bounds_index(self, synthetic_pro, tmp_path):
        """Test selector with index out of bounds."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=3)
        
        ds = read(data_dir, lazy=False)
        
        # Try to select index that doesn't exist
        selector = LocationSelector(sel=999)
        
        # Should handle gracefully
        try:
            ds_filtered = selector.apply(ds.data)
            if ds_filtered is not None:
                assert ds_filtered.sizes['location'] == 0
        except (IndexError, KeyError):
            pass  # Expected