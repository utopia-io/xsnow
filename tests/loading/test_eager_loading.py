# tests/loading/test_eager_loading.py
"""
Tests specific to eager loading mode.
Focus: Eager mode correctness and performance.
"""
import pytest
import numpy as np
from pathlib import Path
from xsnow import read
from xsnow.selectors import TimeSelector, LocationSelector


@pytest.mark.synthetic
class TestEagerBasic:
    """Basic eager loading functionality tests."""
    
    def test_eager_basic(self, synthetic_pro, tmp_path):
        """Test basic eager loading of single file."""
        filepath = synthetic_pro.basic(tmp_path, n_layers=5)
        
        ds = read(filepath, lazy=False)
        
        assert ds is not None
        assert 'height' in ds.data
        assert 'density' in ds.data
        assert ds.sizes['time'] == 1
        assert ds.sizes['location'] == 1
    
    def test_eager_multiple_files(self, synthetic_pro, tmp_path):
        """Test eager loading of multiple files."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=3)
        
        ds = read(data_dir, lazy=False)
        
        assert ds is not None
        assert ds.sizes['location'] == 5
        assert ds.sizes['time'] == 3
    
    def test_eager_single_cpu(self, synthetic_pro, tmp_path):
        """Test eager loading with n_cpus_use=1."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=3)
        
        ds = read(data_dir, lazy=False, n_cpus_use=1)
        
        assert ds is not None
        assert ds.sizes['location'] == 3
    
    def test_eager_multi_cpu(self, synthetic_pro, tmp_path):
        """Test eager loading with multiple CPUs."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        ds = read(data_dir, lazy=False, n_cpus_use=4)
        
        assert ds is not None
        assert ds.sizes['location'] == 5


@pytest.mark.synthetic
class TestEagerWithFilters:
    """Test eager loading with various filters."""
    
    def test_eager_with_time_filter(self, synthetic_pro, tmp_path):
        """Test eager loading with time selector."""
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=[3, 3, 3, 3, 3])
        
        time_selector = TimeSelector(sel=slice('2025-01-02', '2025-01-04'))
        ds = read(filepath, time=time_selector, lazy=False)
        
        assert ds is not None
        assert ds.sizes['time'] == 3  # Days 2, 3, 4
    
    
    def test_eager_with_location_filter(self, synthetic_pro, tmp_path):
        """Test eager loading with location selector."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        # Read all first
        ds_all = read(data_dir, lazy=False)
        
        # Get location mapping to select specific station
        location_names = list(ds_all.location.values)
        target_station = str(location_names[0])
        
        location_selector = LocationSelector(sel=target_station)
        ds_filtered = read(data_dir, location=location_selector, lazy=False)
        
        # Location dimension gets squeezed when selecting single location
        if 'location' in ds_filtered.sizes:
            assert ds_filtered.sizes['location'] == 1
        else:
            # Dimension was squeezed - this is expected
            assert 'location' not in ds_filtered.dims
        
    def test_eager_with_parser_kwargs(self, synthetic_pro, tmp_path):
        """Test eager loading with custom parser arguments."""
        filepath = synthetic_pro.with_soil_layers(tmp_path, n_snow=5, n_soil=3)
        
        # Load with soil removal
        ds_no_soil = read(filepath, lazy=False, remove_soil=True)
        
        # Load with soil
        ds_with_soil = read(filepath, lazy=False, remove_soil=False)
        
        # Dataset with soil should have more valid layers
        height_no_soil = ds_no_soil['height'].values[0, 0, 0, 0, :]
        height_with_soil = ds_with_soil['height'].values[0, 0, 0, 0, :]
        
        valid_no_soil = np.sum(~np.isnan(height_no_soil))
        valid_with_soil = np.sum(~np.isnan(height_with_soil))
        
        assert valid_with_soil > valid_no_soil


@pytest.mark.synthetic
class TestEagerErrorHandling:
    """Test error handling in eager mode."""
    
    def test_eager_empty_directory(self, tmp_path):
        """Test eager loading of empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        ds = read(empty_dir, lazy=False)
        
        assert ds is None
    
    def test_eager_nonexistent_path(self):
        """Test eager loading of nonexistent path."""
        ds = read("/path/that/does/not/exist", lazy=False)
        
        assert ds is None
    
    def test_eager_corrupted_files(self, synthetic_pro, tmp_path):
        """Test eager mode with some corrupted files."""
        # Create mix of valid and corrupted files
        synthetic_pro.basic(tmp_path, station_name="VALID1")
        synthetic_pro.corrupted(tmp_path, corruption_type='malformed_data')
        synthetic_pro.basic(tmp_path, station_name="VALID2")
        
        # Should load valid files and skip corrupted ones
        ds = read(tmp_path, lazy=False)
        
        # Should have 2 valid stations
        assert ds is not None
        assert ds.sizes['location'] >= 2

    def test_eager_mixed_file_types(self, synthetic_pro, tmp_path):
        """Test eager mode ignores non-data files."""
        synthetic_pro.basic(tmp_path, station_name="DATA")
        (tmp_path / "readme.txt").write_text("Not a data file")
        (tmp_path / "image.png").write_bytes(b"fake image")
        
        ds = read(tmp_path, lazy=False)
        
        assert ds is not None
        assert ds.sizes['location'] == 1


@pytest.mark.synthetic
class TestEagerMemoryUsage:
    """Test memory characteristics of eager loading."""
    
    def test_eager_memory_profile(self, synthetic_pro, tmp_path, memory_monitor):
        """Test memory usage during eager loading."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=10, n_profiles_each=5)
        
        memory_monitor.start()
        ds = read(data_dir, lazy=False)
        increase = memory_monitor.increase()
        
        assert ds is not None
        assert increase < 500  # Should use less than 500 MB for this small dataset
        
        memory_monitor.report()
    
    def test_eager_loads_immediately(self, synthetic_pro, tmp_path):
        """Test that eager mode loads all data immediately."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        
        ds = read(data_dir, lazy=False)
        
        # All data should be in memory (no dask arrays)
        for var in ds.data.data_vars:
            assert not hasattr(ds[var].data, 'compute'), f"{var} should not be a dask array"


@pytest.mark.synthetic
class TestEagerSequentialVsParallel:
    """Compare sequential vs parallel eager loading."""
    
    def test_eager_sequential_vs_parallel(self, synthetic_pro, tmp_path, timer):
        """Compare performance of sequential vs parallel eager loading."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=20, n_profiles_each=10)
        
        # Sequential
        timer.start('eager_sequential')
        ds_seq = read(data_dir, lazy=False, n_cpus_use=1)
        timer.stop('eager_sequential')
        
        # Parallel
        timer.start('eager_parallel')
        ds_par = read(data_dir, lazy=False, n_cpus_use=4)
        timer.stop('eager_parallel')
        
        timer.report()
        
        # Both should produce same result
        assert ds_seq.sizes == ds_par.sizes
        
        # Parallel should be faster (with some tolerance for overhead)
        # Note: May not always be true for very small datasets
        seq_time = timer.get('eager_sequential')
        par_time = timer.get('eager_parallel')
        
        print(f"\nSequential: {seq_time:.2f}s, Parallel: {par_time:.2f}s")
        print(f"Speedup: {seq_time/par_time:.2f}x")
    
    def test_eager_cpu_scaling(self, synthetic_pro, tmp_path, timer):
        """Test how eager loading scales with CPU count."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=30)
        
        for n_cpus in [1, 2, 4]:
            timer.start(f'eager_{n_cpus}_cpus')
            ds = read(data_dir, lazy=False, n_cpus_use=n_cpus)
            timer.stop(f'eager_{n_cpus}_cpus')
            
            assert ds is not None
            assert ds.sizes['location'] == 30
        
        timer.report()


@pytest.mark.synthetic
class TestEagerVaryingLayerCounts:
    """Test eager loading with varying layer counts."""
    
    def test_eager_auto_max_layers(self, synthetic_pro, tmp_path):
        """Test automatic max_layers detection in eager mode."""
        layer_counts = [5, 10, 3, 15, 7]
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=layer_counts)
        
        ds = read(filepath, lazy=False, max_layers='auto')
        
        assert ds is not None
        # Should be padded to max(layer_counts) = 15
        assert ds.sizes['layer'] == 15
    
    def test_eager_explicit_max_layers(self, synthetic_pro, tmp_path):
        """Test explicit max_layers specification."""
        filepath = synthetic_pro.varying_layers(tmp_path, layer_counts=[3, 5, 4])
        
        ds = read(filepath, lazy=False, max_layers=20)
        
        assert ds is not None
        assert ds.sizes['layer'] == 20
    
    def test_eager_parallel_max_layers_detection(self, synthetic_pro, tmp_path):
        """Test that parallel max_layers scan works correctly."""
        # Create multiple files with varying layer counts
        for i in range(10):
            synthetic_pro.varying_layers(
                tmp_path,  # Pass directory, not file path
                layer_counts=[2, 5, 8, 3, 10]
            )
        
        ds = read(tmp_path, lazy=False, n_cpus_use=4, max_layers='auto')
        
        assert ds is not None
        assert ds.sizes['layer'] == 10  # max across all files
        

@pytest.mark.synthetic
class TestEagerLoadingWithStringCoords:
    """Test eager loading produces string coordinates."""
    
    def test_eager_mode_creates_string_coords(self, synthetic_pro, tmp_path):
        """Test eager loading creates string location coordinates."""
        from xsnow import read
        
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)
        ds = read(data_dir, lazy=False)
        
        assert ds.data.location.dtype.kind in ('U', 'S', 'O')
        assert 'mapping' not in ds.data.location.attrs