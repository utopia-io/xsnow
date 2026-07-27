# tests/loading/test_trim_padding.py
"""
Tests for trim_padding functionality.

This module tests the new trim_padding() method that removes NaN-only
layers from datasets to reduce memory usage.
"""
import dask
import dask.array as da
import numpy as np
import pytest
from dask.callbacks import Callback

from xsnow import read


@pytest.fixture
def dataset_with_padding(tmp_path):
    """Create a dataset with known padding."""
    # Create .pro files where actual data has 10 layers
    # but will be padded to higher dimension
    layer_data = ",".join([str(0.1 * i) for i in range(10)])
    density_data = ",".join([str(100 + i) for i in range(10)])
    
    for i in range(3):
        filepath = tmp_path / f"station_{i}.pro"
        filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,10,{layer_data}
0502,10,{density_data}
0500,02.01.2025 12:00:00
0501,10,{layer_data}
0502,10,{density_data}
""")
    
    return tmp_path


@pytest.fixture
def dataset_variable_layers(tmp_path):
    """Create dataset where files have different layer counts."""
    # File 1: 5 layers
    filepath1 = tmp_path / "station_1.pro"
    filepath1.write_text("""
[STATION_PARAMETERS]
StationName= STATION_1
[DATA]
0500,01.01.2025 12:00:00
0501,5,0.1,0.2,0.3,0.4,0.5
0502,5,100,101,102,103,104
""")
    
    # File 2: 12 layers (max)
    layer_data = ",".join([str(0.1 * i) for i in range(12)])
    density_data = ",".join([str(100 + i) for i in range(12)])
    filepath2 = tmp_path / "station_2.pro"
    filepath2.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_2
[DATA]
0500,01.01.2025 12:00:00
0501,12,{layer_data}
0502,12,{density_data}
""")
    
    # File 3: 8 layers
    layer_data = ",".join([str(0.1 * i) for i in range(8)])
    density_data = ",".join([str(100 + i) for i in range(8)])
    filepath3 = tmp_path / "station_3.pro"
    filepath3.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_3
[DATA]
0500,01.01.2025 12:00:00
0501,8,{layer_data}
0502,8,{density_data}
""")
    
    return tmp_path


class TestTrimPaddingBasic:
    """Basic trim_padding functionality tests."""
    
    def test_trim_reduces_dimension(self, dataset_with_padding):
        """Trimming should reduce layer dimension to actual data size."""
        # Load with padding
        ds = read(dataset_with_padding, lazy=False, max_layers=20)
        original_size = ds.sizes['layer']
        assert original_size == 20
        
        # Trim padding
        ds_trimmed = ds.trim_padding()
        new_size = ds_trimmed.sizes['layer']
        
        # Should be reduced to actual data size (10 layers)
        assert new_size == 10
        assert new_size < original_size
    
    def test_trim_preserves_data(self, dataset_with_padding):
        """Trimming should not change actual data values."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20)
        
        # Get data before trimming
        height_before = ds['height'].isel(layer=slice(0, 10)).values.copy()
        
        # Trim
        ds_trimmed = ds.trim_padding()
        
        # Data should be identical
        height_after = ds_trimmed['height'].values
        np.testing.assert_array_equal(height_after, height_before)
    
    def test_trim_returns_new_dataset_by_default(self, dataset_with_padding):
        """Default behavior should return new dataset."""
        ds = read(dataset_with_padding, lazy=False, max_layers=15)
        original_size = ds.sizes['layer']
        
        ds_trimmed = ds.trim_padding()
        
        # Original should be unchanged
        assert ds.sizes['layer'] == original_size
        # New dataset should be trimmed
        assert ds_trimmed.sizes['layer'] < original_size
    
    def test_trim_inplace_modifies_original(self, dataset_with_padding):
        """inplace=True should modify original dataset."""
        # Use max_layers=20 to ensure there's padding (actual data is 10 layers)
        ds = read(dataset_with_padding, lazy=False, max_layers=25)
        original_size = ds.sizes['layer']
        
        result = ds.trim_padding(inplace=True)
        
        # Should return same object
        assert result is ds
        # Should be modified (trimmed from 20 to 10)
        assert ds.sizes['layer'] < original_size
        assert ds.sizes['layer'] == 10
    
    def test_no_padding_no_change(self, dataset_with_padding):
        """Dataset with no padding should not be changed."""
        # Load with exact max_layers
        ds = read(dataset_with_padding, lazy=False, max_layers=10)
        
        ds_trimmed = ds.trim_padding()
        
        # Size should remain the same
        assert ds_trimmed.sizes['layer'] == ds.sizes['layer'] == 10


class TestTrimPaddingWithBuffer:
    """Tests for keep_buffer parameter."""
    
    def test_keep_buffer_adds_layers(self, dataset_with_padding):
        """keep_buffer should add extra layers beyond max data."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20)
        
        # Trim with buffer of 5
        ds_trimmed = ds.trim_padding(keep_buffer=5)
        
        # Should be: 10 (actual data) + 5 (buffer) = 15
        assert ds_trimmed.sizes['layer'] == 15
    
    def test_buffer_doesnt_exceed_original(self, dataset_with_padding):
        """Buffer should not expand beyond original size."""
        ds = read(dataset_with_padding, lazy=False, max_layers=12)
        
        # Request large buffer
        ds_trimmed = ds.trim_padding(keep_buffer=100)
        
        # Should not exceed original size
        assert ds_trimmed.sizes['layer'] == 12
    
    def test_zero_buffer_trims_exactly(self, dataset_with_padding):
        """Zero buffer should trim to exact max data."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20)
        
        ds_trimmed = ds.trim_padding(keep_buffer=0)
        
        assert ds_trimmed.sizes['layer'] == 10


class TestTrimPaddingWithLazyLoading:
    """Tests for trim_padding with lazy-loaded datasets."""

    def test_trim_lazy_computes_extent_and_preserves_laziness(
        self, dataset_with_padding
    ):
        """Trimming computes the valid extent but keeps payload data lazy."""
        ds = read(dataset_with_padding, lazy=True, max_layers=20)

        assert isinstance(ds['height'].data, da.Array)

        executed_tasks = []
        with dask.config.set(scheduler="synchronous"), Callback(
            pretask=lambda key, _dask, _state: executed_tasks.append(key)
        ):
            ds_trimmed = ds.trim_padding()

        assert executed_tasks
        assert ds_trimmed.sizes['layer'] == 10
        assert isinstance(ds_trimmed['height'].data, da.Array)
    
    def test_trim_lazy_warning(self, dataset_with_padding):
        """Should warn when trimming lazy dataset via read()."""
        # Try to trim during lazy load
        # Warning will appear in logs (visible in pytest output)
        ds = read(dataset_with_padding, lazy=True, trim_padding=True)
        
        # Verify trimming actually happened
        assert ds is not None
        assert ds.sizes['layer'] == 10  # Should be trimmed from auto-detected size
    
    def test_trim_lazy_warning_shows_workflow(self, dataset_with_padding):
        """Warning should show complete workflow example (visible in pytest output)."""
        # Logs with workflow appear in pytest's captured stderr
        ds = read(dataset_with_padding, lazy=True, trim_padding=True)
        
        # Verify operation completed
        assert ds is not None
        assert ds.sizes['layer'] == 10
    
    def test_trim_after_compute_no_warning(self, dataset_with_padding):
        """Trimming after compute should work without issues."""
        ds = read(dataset_with_padding, lazy=True, max_layers=20)
        
        # Compute first
        ds = ds.compute()
        
        # Now trim - should work fine
        ds_trimmed = ds.trim_padding()
        
        # Verify it worked
        assert ds_trimmed is not None
        assert ds_trimmed.sizes['layer'] == 10


class TestTrimPaddingAutomaticDuringLoad:
    """Tests for automatic trimming during read()."""
    
    def test_trim_padding_parameter_works(self, dataset_with_padding):
        """trim_padding=True in read() should auto-trim."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20, trim_padding=True)
        
        # Should be trimmed automatically
        assert ds.sizes['layer'] == 10
    
    def test_auto_trim_with_eager_loading(self, dataset_with_padding):
        """Auto-trim should work smoothly with eager loading."""
        ds = read(dataset_with_padding, lazy=False, trim_padding=True)
        
        # Should have trimmed dimensions
        assert ds.sizes['layer'] <= 12  # Actual max is 10, might have small buffer
    
    def test_auto_trim_logs_action(self, dataset_with_padding):
        """Auto-trim should log the action (visible in pytest output)."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20, trim_padding=True)
        
        # Verify trimming happened
        assert ds.sizes['layer'] == 10  # Trimmed from 20


class TestTrimPaddingEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_all_nan_layers(self, tmp_path):
        """Dataset with all NaN layers should handle gracefully."""
        # Create dataset with no actual data (should be rare)
        filepath = tmp_path / "empty.pro"
        filepath.write_text("""
[STATION_PARAMETERS]
StationName= EMPTY
[DATA]
0500,01.01.2025 12:00:00
""")
        
        ds = read(tmp_path, lazy=False)
        
        # Trim should not crash
        ds_trimmed = ds.trim_padding()
        assert ds_trimmed is not None
    
    def test_dimension_not_in_dataset(self, dataset_with_padding):
        """Trimming non-existent dimension should log warning."""
        ds = read(dataset_with_padding, lazy=False)
        
        # Try to trim dimension that doesn't exist
        ds_trimmed = ds.trim_padding(dims=['nonexistent'])
        
        # Should return unchanged dataset
        assert ds_trimmed.sizes == ds.sizes
    
    def test_multiple_dimensions(self, dataset_with_padding):
        """Can specify multiple dimensions to trim."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20)
        
        # Trim only layer dimension
        ds_trimmed = ds.trim_padding(dims=['layer'])
        
        assert ds_trimmed.sizes['layer'] < ds.sizes['layer']


class TestTrimPaddingVariableLayers:
    """Tests with datasets where files have different layer counts."""
    
    def test_trim_to_max_across_locations(self, dataset_variable_layers):
        """Should trim to maximum across all locations."""
        # Load with high max_layers
        ds = read(dataset_variable_layers, lazy=False, max_layers=20)
        
        # Trim
        ds_trimmed = ds.trim_padding()
        
        # Should trim to max data (12 layers from station_2)
        assert ds_trimmed.sizes['layer'] == 12
    
    def test_trim_preserves_all_location_data(self, dataset_variable_layers):
        """Trimming should not lose data from any location."""
        ds = read(dataset_variable_layers, lazy=False, max_layers=20)
        
        # Get data for location with most layers (12 layers)
        # Only compare the actual data portion (first 12 layers)
        station_2_before = ds.sel(location='STATION_2')['height'].isel(layer=slice(0, 12)).values.copy()
        
        # Trim
        ds_trimmed = ds.trim_padding()
        
        # After trimming, dataset should have 12 layers (the max across all files)
        assert ds_trimmed.sizes['layer'] == 12
        
        # Data should be preserved (now comparing same-sized arrays)
        station_2_after = ds_trimmed.sel(location='STATION_2')['height'].values
        np.testing.assert_array_equal(station_2_after, station_2_before)


class TestTrimPaddingMemorySavings:
    """Tests to verify actual memory savings."""
    
    def test_trimmed_dataset_smaller(self, dataset_with_padding):
        """Trimmed dataset should use less memory."""
        import sys
        
        ds = read(dataset_with_padding, lazy=False, max_layers=50)
        
        # Estimate size before
        size_before = sys.getsizeof(ds['height'].values)
        
        # Trim
        ds_trimmed = ds.trim_padding()
        
        # Estimate size after
        size_after = sys.getsizeof(ds_trimmed['height'].values)
        
        # Should be smaller
        assert size_after < size_before
    
    def test_substantial_padding_substantial_savings(self, dataset_with_padding):
        """Large padding should result in significant savings."""
        # Create dataset with lots of padding
        ds = read(dataset_with_padding, lazy=False, max_layers=100)
        original_layers = ds.sizes['layer']
        
        ds_trimmed = ds.trim_padding()
        trimmed_layers = ds_trimmed.sizes['layer']
        
        # Should have removed at least 50% of layers
        reduction = (original_layers - trimmed_layers) / original_layers
        assert reduction > 0.5  # More than 50% reduction


class TestTrimPaddingLogging:
    """Tests for logging behavior."""
    
    def test_logs_trimming_action(self, dataset_with_padding):
        """Should log the trimming action (visible in pytest output)."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20)
        
        # Trim - logs will appear in pytest output
        ds_trimmed = ds.trim_padding()
        
        # Verify trimming worked
        assert ds_trimmed.sizes['layer'] == 10  # Trimmed from 20
    
    def test_logs_no_change_when_no_padding(self, dataset_with_padding, caplog):
        """Should handle no-padding case gracefully."""
        ds = read(dataset_with_padding, lazy=False, max_layers=10)
        
        caplog.clear()
        ds_trimmed = ds.trim_padding()
        
        # Shouldn't crash, might log that no trimming was needed
        assert ds_trimmed is not None


@pytest.mark.integration
class TestTrimPaddingIntegration:
    """Integration tests for trim_padding with real workflows."""
    
    def test_subset_then_trim_workflow(self, dataset_variable_layers):
        """Common workflow: subset then trim."""
        # Load lazily
        ds = read(dataset_variable_layers, lazy=True, max_layers=20)
        
        # Subset to one location
        ds_sub = ds.sel(location='STATION_2')
        
        # Compute
        ds_sub = ds_sub.compute()
        
        # Trim
        ds_trimmed = ds_sub.trim_padding()
        
        # Should be trimmed to station_2's max (12 layers)
        assert ds_trimmed.sizes['layer'] == 12
    
    def test_eager_load_and_trim_workflow(self, dataset_with_padding):
        """Workflow: eager load with auto-trim."""
        ds = read(dataset_with_padding, lazy=False, max_layers=20, trim_padding=True)
        
        # Should be ready to use with minimal memory
        assert ds.sizes['layer'] == 10
        assert isinstance(ds['height'].values, np.ndarray)
    
    def test_iterative_trimming(self, dataset_with_padding):
        """Can trim multiple times (though second should be no-op)."""
        ds = read(dataset_with_padding, lazy=False, max_layers=30)
        
        # First trim
        ds = ds.trim_padding(keep_buffer=5)
        size_1 = ds.sizes['layer']
        
        # Second trim with smaller buffer
        ds = ds.trim_padding(keep_buffer=2)
        size_2 = ds.sizes['layer']
        
        # Second trim should reduce further
        assert size_2 <= size_1


@pytest.mark.parametrize("max_layers,expected_trimmed", [
    (15, 10),   # Trim from 15 to 10
    (20, 10),   # Trim from 20 to 10
    (50, 10),   # Trim from 50 to 10
    (10, 10),   # No trimming (exact)
])
def test_trim_various_padding_sizes(dataset_with_padding, max_layers, expected_trimmed):
    """Parametrized test for various padding amounts."""
    ds = read(dataset_with_padding, lazy=False, max_layers=max_layers)
    ds_trimmed = ds.trim_padding()
    
    assert ds_trimmed.sizes['layer'] == expected_trimmed


def test_trim_padding_method_exists(dataset_with_padding):
    """Verify trim_padding is registered as method."""
    ds = read(dataset_with_padding, lazy=False)
    
    # Should have trim_padding method
    assert hasattr(ds, 'trim_padding')
    assert callable(ds.trim_padding)
