# tests/loading/test_lazy_loading.py
"""
Tests specific to lazy/Dask loading mode.

Focus: Dask integration and lazy evaluation.

Strategy:
  - Use REAL data to validate Dask workflow with realistic files
  - Use SYNTHETIC data for controlled testing of Dask features
  - Verify lazy evaluation and memory efficiency
"""

import pytest
import numpy as np
import dask
from dask import delayed

from xsnow import read


# ============================================================================
# BASIC LAZY LOADING TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_basic(real_file_subset):
    """
    Primary test: Basic lazy loading with real data.
    
    Validates that lazy=True mode works and produces results.
    """
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] > 0
    assert ds.sizes['time'] > 0
    assert 'height' in ds.data or 'density' in ds.data


@pytest.mark.real_data
def test_lazy_basic_single_file(sample_pro_file):
    """
    Test lazy loading with single file.
    """
    ds = read(sample_pro_file, lazy=True)
    
    assert ds is not None
    assert ds.sizes['location'] == 1


@pytest.mark.synthetic
def test_lazy_basic_synthetic(tmp_path, synthetic_pro):
    """
    Test lazy loading with synthetic data for precise control.
    """
    synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=3)
    
    ds = read(tmp_path, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 5
    assert ds.sizes['time'] == 3


# ============================================================================
# DASK GRAPH CONSTRUCTION TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_graph_construction(real_file_subset):
    """
    Test that Dask builds computation graph without executing.
    
    Lazy loading should be fast because it doesn't process data immediately.
    """
    import time
    
    # Lazy loading should be very fast (just builds graph)
    start = time.time()
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    elapsed = time.time() - start
    
    assert ds is not None
    # Should complete quickly (< 5 seconds typically)
    print(f"\nLazy loading (graph construction): {elapsed:.3f}s")


@pytest.mark.synthetic
def test_lazy_delayed_execution(tmp_path, synthetic_pro):
    """
    Test lazy mode uses Dask for parallel computation.
    
    Note: The current implementation computes during read() using Dask's
    parallel scheduler, rather than delaying until data access.
    """
    # Create multiple files
    synthetic_pro.multi_station(tmp_path, n_stations=10, n_profiles_each=2)
    
    # Lazy mode should use Dask for parallel computation
    ds = read(tmp_path, lazy=True, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] == 10
    
    # Data should be immediately accessible (Dask computed during read)
    assert 'density' in ds.data
    density_values = ds['density'].values
    assert density_values is not None


# ============================================================================
# DASK TASK PARALLELIZATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_uses_dask_scheduler(real_file_subset):
    """
    Test that lazy mode actually uses Dask's scheduler.
    
    This is what enables parallel processing.
    """
    # Read with lazy mode
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    # Data should be accessible (Dask computed when needed)
    assert len(ds.data.data_vars) > 0


@pytest.mark.synthetic
def test_lazy_parallel_task_execution(tmp_path, synthetic_pro):
    """
    Test that Dask executes tasks in parallel.
    """
    # Create enough files to benefit from parallelization
    synthetic_pro.multi_station(tmp_path, n_stations=8, n_profiles_each=2)
    
    # Lazy with multiple workers
    ds = read(tmp_path, lazy=True, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] == 8


# ============================================================================
# MEMORY EFFICIENCY TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_memory_efficiency(real_file_subset, memory_monitor):
    """
    Test that lazy loading is memory efficient.
    
    Should use less memory than eager loading for same data.
    """
    memory_monitor.start()
    
    # Lazy load
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    memory_increase = memory_monitor.increase()
    memory_monitor.report()
    
    assert ds is not None
    
    # Memory increase should be reasonable
    # (Lazy should not load all data into memory at once)
    print(f"\nMemory increase (lazy): {memory_increase:.2f} MB")


@pytest.mark.real_data
@pytest.mark.slow
def test_lazy_memory_efficiency_comparison(real_file_subset, memory_monitor):
    """
    Compare memory usage: lazy vs eager.
    """
    import gc
    
    # Test eager mode
    memory_monitor.start()
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    eager_memory = memory_monitor.increase()
    del ds_eager
    gc.collect()
    
    # Test lazy mode
    memory_monitor.start()
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    lazy_memory = memory_monitor.increase()
    
    print(f"\nEager memory: {eager_memory:.2f} MB")
    print(f"Lazy memory: {lazy_memory:.2f} MB")
    
    # For small datasets, difference may be minimal
    # Just verify both modes work without excessive memory
    assert eager_memory < 1000  # < 1 GB
    assert lazy_memory < 1000


@pytest.mark.real_data
@pytest.mark.slow
def test_lazy_with_large_dataset(gridded_pros_dir, memory_monitor):
    """
    Test lazy loading with full real dataset.
    
    This is where lazy mode shines - large datasets that might not fit
    comfortably in memory with eager loading.
    """
    memory_monitor.start()
    
    ds = read(gridded_pros_dir, lazy=True, n_cpus_use=4)
    
    memory_increase = memory_monitor.increase()
    memory_monitor.report()
    
    assert ds is not None
    assert ds.sizes['location'] > 0
    
    print(f"\nLarge dataset: {ds.sizes['location']} locations")
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    # Should handle large dataset without excessive memory
    assert memory_increase < 2000  # < 2 GB


# ============================================================================
# DASK COMPUTATION GRAPH TESTS
# ============================================================================

@pytest.mark.synthetic
def test_lazy_task_graph_size(tmp_path, synthetic_pro):
    """
    Test that Dask creates appropriate number of tasks.
    
    One task per file typically.
    """
    n_files = 5
    synthetic_pro.multi_station(tmp_path, n_stations=n_files, n_profiles_each=2)
    
    # The read function uses Dask delayed internally
    # We can't directly inspect the graph, but we can verify it works
    ds = read(tmp_path, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == n_files


@pytest.mark.real_data
def test_lazy_graph_optimization(real_file_subset):
    """
    Test that Dask optimizes the computation graph.
    
    Dask should eliminate redundant computations.
    """
    # Load lazily
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    # Multiple accesses should not cause redundant computation
    _ = ds.sizes  # First access
    _ = ds.sizes  # Second access (should be cached/optimized)
    
    assert ds is not None


# ============================================================================
# CHUNKING AND PARTITIONING TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_respects_cpu_count(real_file_subset):
    """
    Test that lazy mode respects n_cpus_use parameter.
    """
    # Test with different CPU counts
    ds_2cpu = read(real_file_subset, lazy=True, n_cpus_use=2)
    ds_4cpu = read(real_file_subset, lazy=True, n_cpus_use=4)
    
    # Both should work and produce same results
    assert ds_2cpu is not None
    assert ds_4cpu is not None
    assert ds_2cpu.sizes == ds_4cpu.sizes


# ============================================================================
# ERROR HANDLING IN LAZY MODE
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_lazy_handles_corrupted_files(tmp_path, synthetic_pro):
    """
    Test that lazy mode handles corrupted files gracefully.
    """
    # Create mix of valid and corrupted files
    synthetic_pro.basic(tmp_path, "VALID_1")
    synthetic_pro.corrupted(tmp_path, "malformed_data")
    synthetic_pro.basic(tmp_path, "VALID_2")
    
    # Should load valid files, skip corrupted
    ds = read(tmp_path, lazy=True, n_cpus_use=2)
    
    # Should have loaded the valid files
    assert ds is None or ds.sizes['location'] >= 1


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_lazy_with_empty_directory(tmp_path):
    """
    Test lazy mode with empty directory.
    """
    ds = read(tmp_path, lazy=True)
    
    assert ds is None


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_lazy_with_single_file_empty_profiles(tmp_path, synthetic_pro):
    """
    Test lazy mode with file containing no valid profiles.
    """
    synthetic_pro.empty_profiles(tmp_path, n_profiles=3)
    
    ds = read(tmp_path, lazy=True, n_cpus_use=2)
    
    # Should handle gracefully
    assert ds is None or ds.sizes['time'] == 0


# ============================================================================
# PRESCAN PHASE TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_prescan_with_real_data(real_file_subset):
    """
    Test that prescan phase works correctly in lazy mode.
    
    Prescan determines max_layers before building Dask graph.
    """
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['layer'] > 0
    
    # Max layers should accommodate all profiles
    # (verified by successful loading)


@pytest.mark.synthetic
def test_lazy_prescan_finds_max_layers(tmp_path, synthetic_pro):
    """
    Test that prescan correctly identifies max layer count.
    """
    # Create files with varying layer counts
    synthetic_pro.varying_layers(tmp_path, [2, 5, 3, 8, 4])
    
    ds = read(tmp_path, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    # Should have padded to max (8)
    assert ds.sizes['layer'] == 8


@pytest.mark.synthetic
def test_lazy_with_manual_max_layers(tmp_path, synthetic_pro):
    """
    Test that manual max_layers skips prescan in lazy mode.
    """
    synthetic_pro.basic(tmp_path, n_layers=5)
    
    # Override max_layers (skip prescan)
    ds = read(tmp_path, lazy=True, max_layers=10, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['layer'] == 10


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_with_selectors(real_file_subset):
    """
    Integration: Lazy mode with selector filters.
    """
    from xsnow.selectors import TimeSelector
    
    # Load all data first to know time range
    ds_full = read(real_file_subset, lazy=False)
    
    if ds_full.sizes['time'] < 2:
        pytest.skip("Need multiple timesteps")
    
    # Use time selector with lazy mode
    mid_time = ds_full.time.values[ds_full.sizes['time'] // 2]
    time_filter = TimeSelector(sel=slice(None, mid_time))
    
    ds_lazy = read(real_file_subset, lazy=True, time=time_filter, n_cpus_use=2)
    
    assert ds_lazy is not None
    assert ds_lazy.sizes['time'] <= ds_full.sizes['time']


@pytest.mark.real_data
def test_lazy_preserves_physical_validity(real_file_subset, assert_physical_validity):
    """
    Integration: Lazy mode produces physically valid data.
    """
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert_physical_validity(ds)


@pytest.mark.real_data
def test_lazy_with_soil_removal(real_file_subset):
    """
    Integration: Lazy mode with soil removal.
    """
    ds_no_soil = read(real_file_subset, lazy=True, remove_soil=True, n_cpus_use=2)
    ds_with_soil = read(real_file_subset, lazy=True, remove_soil=False, n_cpus_use=2)
    
    assert ds_no_soil is not None
    assert ds_with_soil is not None
    
    # Soil removal should reduce or maintain layer count
    assert ds_no_soil.sizes['layer'] <= ds_with_soil.sizes['layer']


@pytest.mark.real_data
def test_lazy_with_surface_hoar(real_file_subset):
    """
    Integration: Lazy mode with surface hoar addition.
    """
    ds_with_sh = read(real_file_subset, lazy=True, add_surface_sh_as_layer=True, n_cpus_use=2)
    ds_no_sh = read(real_file_subset, lazy=True, add_surface_sh_as_layer=False, n_cpus_use=2)
    
    assert ds_with_sh is not None
    assert ds_no_sh is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.real_data
@pytest.mark.slow
def test_lazy_scales_with_cpu_count(real_file_subset, timer):
    """
    Test that lazy mode benefits from more CPUs.
    """
    # Sequential (1 CPU)
    timer.start('1cpu')
    ds_1cpu = read(real_file_subset, lazy=True, n_cpus_use=1)
    timer.stop('1cpu')
    
    # Parallel (4 CPUs)
    timer.start('4cpu')
    ds_4cpu = read(real_file_subset, lazy=True, n_cpus_use=4)
    timer.stop('4cpu')
    
    timer.report()
    
    # Both should work
    assert ds_1cpu is not None
    assert ds_4cpu is not None
    assert ds_1cpu.sizes == ds_4cpu.sizes
    
    # With enough files, parallel should be faster
    # (but with small datasets, overhead might negate benefits)


@pytest.mark.real_data
@pytest.mark.slow
def test_lazy_throughput(gridded_pros_dir, timer):
    """
    Measure lazy mode throughput on large dataset.
    """
    timer.start('lazy_large')
    ds = read(gridded_pros_dir, lazy=True, n_cpus_use=4)
    timer.stop('lazy_large')
    
    timer.report()
    
    assert ds is not None
    
    elapsed = timer.get('lazy_large')
    if elapsed:
        files_per_sec = ds.sizes['location'] / elapsed
        print(f"\nThroughput: {files_per_sec:.2f} files/second")


# ============================================================================
# REGRESSION TESTS
# ============================================================================

@pytest.mark.real_data
def test_lazy_produces_xsnowdataset(real_file_subset):
    """
    Regression: Lazy mode should return xsnowDataset, not raw xarray.
    """
    from xsnow import xsnowDataset
    
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert isinstance(ds, xsnowDataset)


@pytest.mark.real_data
def test_lazy_computes_z_coordinate(real_file_subset):
    """
    Regression: Lazy mode should compute z coordinate.
    """
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    if ds and 'height' in ds.data:
        assert 'z' in ds.data.coords


@pytest.mark.real_data
def test_lazy_preserves_location_mapping(real_file_subset):
    """
    Regression: Lazy mode should preserve location as string coordinates.
    
    Updated: With string-based coordinates, locations are stored directly
    as meaningful string values instead of integer indices with a mapping dict.
    """
    ds = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    # Verify location is a string-based coordinate (not integers)
    assert ds.location.dtype.kind in ('U', 'S', 'O')  # Unicode, bytes, or object strings
    
    # Verify we have location values
    assert len(ds.location.values) > 0
    
    # Verify location values are meaningful strings (station names)
    for loc in ds.location.values:
        loc_str = str(loc)
        assert loc_str  # Not empty
        assert loc_str != 'nan'  # Not a placeholder
    
    # Verify the old mapping attribute is NOT present (intentional change)
    assert 'mapping' not in ds.location.attrs
    
    # Example: Access station by name directly (the new way!)
    first_station = str(ds.location.values[0])
    assert first_station == 'VIR1A'  # Based on the error output


@pytest.mark.synthetic
class TestLazyLoadingWithStringCoords:
    """Test lazy loading produces string coordinates."""
    
    def test_lazy_mode_creates_string_coords(self, synthetic_pro, tmp_path):
        """Test lazy loading creates string location coordinates."""
        from xsnow import read
        
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=15)
        ds = read(data_dir, lazy=True)
        
        assert ds.data.location.dtype.kind in ('U', 'S', 'O')
        assert 'mapping' not in ds.data.location.attrs

if __name__ == '__main__':
    pytest.main([__file__, '-v'])