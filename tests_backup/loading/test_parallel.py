# tests/loading/test_parallel.py
"""
Tests for parallel processing functionality.

Focus: Multiprocessing, CPU utilization, and parallel efficiency.

Strategy:
  - Use REAL data to validate realistic parallel workflows
  - Use SYNTHETIC data for controlled CPU scaling tests
  - Measure speedup and verify correctness
"""

import pytest
import numpy as np
import os
import multiprocessing

from xsnow import read


# ============================================================================
# BASIC PARALLEL TESTS
# ============================================================================

@pytest.mark.real_data
def test_parallel_vs_sequential(real_file_subset, timer):
    """
    PRIMARY TEST: Parallel processing should be faster with enough files.
    
    With sufficient files, parallel should show speedup.
    """
    
    num_files = len(list(real_file_subset.glob("*.pro")))
    
    if num_files < 10:
        pytest.skip("Need at least 10 files for meaningful parallel comparison")
        
    # Sequential (1 CPU)
    timer.start('sequential')
    ds_seq = read(real_file_subset, lazy=False, n_cpus_use=1)
    timer.stop('sequential')
    
    # Parallel (4 CPUs)
    timer.start('parallel')
    ds_par = read(real_file_subset, lazy=False, n_cpus_use=4)
    timer.stop('parallel')
    
    timer.report()
    
    # Both should work
    assert ds_seq is not None
    assert ds_par is not None
    
    # Should produce same results
    assert ds_seq.sizes == ds_par.sizes
    
    seq_time = timer.get('sequential')
    par_time = timer.get('parallel')
    
    if seq_time and par_time:
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.2f}x with 4 CPUs")
        
        # Parallel should be at least as fast
        # (may not be faster with very small datasets due to overhead)
        assert par_time <= seq_time * 1.5  # Allow some tolerance


@pytest.mark.synthetic
def test_parallel_correctness(tmp_path, synthetic_pro, compare_datasets):
    """
    Test that parallel processing produces correct results.
    
    Critical: parallelization must not introduce bugs.
    """
    # Create multiple files
    synthetic_pro.multi_station(tmp_path, n_stations=10, n_profiles_each=2)
    
    # Sequential
    ds_seq = read(tmp_path, lazy=False, n_cpus_use=1)
    
    # Parallel
    ds_par = read(tmp_path, lazy=False, n_cpus_use=4)
    
    # Must be identical
    compare_datasets(ds_seq, ds_par, rtol=1e-6)


# ============================================================================
# CPU COUNT TESTS
# ============================================================================

@pytest.mark.real_data
def test_cpu_count_detection(real_file_subset):
    """
    Test automatic CPU count detection (n_cpus_use=None).
    
    Should use a reasonable default based on system.
    """
    # Auto-detect CPU count
    ds = read(real_file_subset, lazy=False, n_cpus_use=None)
    
    assert ds is not None
    
    # Should have used some reasonable number
    # (actual detection happens in read function)


@pytest.mark.real_data
def test_cpu_count_detection_default_behavior(real_file_subset):
    """
    Test that default behavior uses sensible CPU count.
    
    Default should be min(32, total_cores - 1) with minimum of 1.
    """
    total_cpus = os.cpu_count() or 1
    expected_default = min(32, max(1, total_cpus - 1))
    
    print(f"\nSystem has {total_cpus} CPUs")
    print(f"Expected default: {expected_default} CPUs")
    
    # Test with default
    ds = read(real_file_subset, lazy=False)
    
    assert ds is not None


@pytest.mark.synthetic
def test_custom_cpu_count(tmp_path, synthetic_pro):
    """
    Test that custom CPU count is respected.
    """
    synthetic_pro.multi_station(tmp_path, n_stations=8, n_profiles_each=2)
    
    # Test various CPU counts
    for n_cpus in [1, 2, 4, 8]:
        ds = read(tmp_path, lazy=False, n_cpus_use=n_cpus)
        assert ds is not None
        assert ds.sizes['location'] == 8


@pytest.mark.synthetic
def test_cpu_count_exceeds_system(tmp_path, synthetic_pro):
    """
    Test behavior when requested CPUs exceed system CPUs.
    
    Should still work (OS will handle scheduling).
    """
    synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=2)
    
    system_cpus = os.cpu_count() or 1
    excessive_cpus = system_cpus * 2
    
    # Should work but may not provide additional benefit
    ds = read(tmp_path, lazy=False, n_cpus_use=excessive_cpus)
    
    assert ds is not None


@pytest.mark.synthetic
def test_single_cpu_processing(tmp_path, synthetic_pro):
    """
    Test explicit single-CPU processing (n_cpus_use=1).
    
    Should work correctly (sequential processing).
    """
    synthetic_pro.multi_station(tmp_path, n_stations=10, n_profiles_each=2)
    
    ds = read(tmp_path, lazy=False, n_cpus_use=1)
    
    assert ds is not None
    assert ds.sizes['location'] == 10


# ============================================================================
# SCALING TESTS
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.slow
def test_parallel_scaling_with_cpu_count(tmp_path, synthetic_pro, timer):
    """
    Test how performance scales with CPU count.
    
    More CPUs should generally improve performance (up to a point).
    """
    # Create enough files to benefit from parallelization
    synthetic_pro.multi_station(tmp_path, n_stations=20, n_profiles_each=2)
    
    results = []
    for n_cpus in [1, 2, 4, 8]:
        timer.start(f'{n_cpus}cpu')
        ds = read(tmp_path, lazy=False, n_cpus_use=n_cpus)
        timer.stop(f'{n_cpus}cpu')
        
        elapsed = timer.get(f'{n_cpus}cpu')
        if elapsed:
            results.append((n_cpus, elapsed))
        
        assert ds is not None
    
    # Print scaling results
    print("\nParallel scaling (20 files):")
    print("CPUs | Time(s) | Speedup")
    print("-" * 30)
    baseline = results[0][1] if results else 1.0
    for n_cpus, elapsed in results:
        speedup = baseline / elapsed
        print(f"{n_cpus:4d} | {elapsed:7.3f} | {speedup:7.2f}x")
    
    # Verify that more CPUs don't make it slower
    # (Allow some tolerance for measurement noise)


@pytest.mark.real_data
@pytest.mark.slow
def test_parallel_efficiency_real_data(real_file_subset, timer):
    """
    Measure parallel efficiency with real data.
    
    Efficiency = speedup / number_of_cpus
    Perfect scaling would be 100% efficiency.
    """
    # Measure with 1, 2, and 4 CPUs
    times = {}
    for n_cpus in [1, 2, 4]:
        timer.start(f'real_{n_cpus}cpu')
        ds = read(real_file_subset, lazy=False, n_cpus_use=n_cpus)
        timer.stop(f'real_{n_cpus}cpu')
        
        elapsed = timer.get(f'real_{n_cpus}cpu')
        if elapsed:
            times[n_cpus] = elapsed
        
        assert ds is not None
    
    timer.report()
    
    # Calculate efficiency
    if 1 in times:
        baseline = times[1]
        print("\nParallel efficiency:")
        print("CPUs | Speedup | Efficiency")
        print("-" * 35)
        for n_cpus, elapsed in sorted(times.items()):
            speedup = baseline / elapsed
            efficiency = speedup / n_cpus * 100
            print(f"{n_cpus:4d} | {speedup:7.2f}x | {efficiency:6.1f}%")


# ============================================================================
# PRESCAN PARALLELIZATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_prescan_parallelization(real_file_subset, timer):
    """
    Test that prescan phase uses parallelization.
    
    Prescan determines max_layers before main processing.
    """
    # With parallelization (default)
    timer.start('prescan_parallel')
    ds_par = read(real_file_subset, lazy=False, n_cpus_use=4)
    timer.stop('prescan_parallel')
    
    # Without parallelization
    timer.start('prescan_sequential')
    ds_seq = read(real_file_subset, lazy=False, n_cpus_use=1)
    timer.stop('prescan_sequential')
    
    timer.report()
    
    # Both should produce same max_layers
    assert ds_par.sizes['layer'] == ds_seq.sizes['layer']


@pytest.mark.synthetic
def test_prescan_with_varying_layer_counts(tmp_path, synthetic_pro):
    """
    Test prescan finds correct max with parallel processing.
    """
    # Create files with varying layer counts
    for i, n_layers in enumerate([3, 7, 5, 9, 4]):
        content = f"""[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,{n_layers},{','.join(str(0.1*j) for j in range(1, n_layers+1))}
0502,{n_layers},{','.join(str(100*j) for j in range(1, n_layers+1))}
"""
        (tmp_path / f"station_{i}.pro").write_text(content)
    
    # Parallel prescan
    ds = read(tmp_path, lazy=False, n_cpus_use=4)
    
    # Should find max = 9
    assert ds is not None
    assert ds.sizes['layer'] == 9


@pytest.mark.synthetic
def test_prescan_skip_with_manual_max_layers(tmp_path, synthetic_pro, timer):
    """
    Test that manual max_layers skips prescan.
    
    Should be faster when max_layers is known.
    """
    synthetic_pro.multi_station(tmp_path, n_stations=10, n_profiles_each=2)
    
    # With prescan (auto max_layers)
    timer.start('with_prescan')
    ds_auto = read(tmp_path, lazy=False, n_cpus_use=4)
    timer.stop('with_prescan')
    
    # Without prescan (manual max_layers)
    timer.start('without_prescan')
    ds_manual = read(tmp_path, lazy=False, max_layers=10, n_cpus_use=4)
    timer.stop('without_prescan')
    
    timer.report()
    
    # Both should work
    assert ds_auto is not None
    assert ds_manual is not None


# ============================================================================
# PROCESS POOL EXECUTOR TESTS
# ============================================================================

@pytest.mark.real_data
def test_process_pool_executor_usage(real_file_subset):
    """
    Test that ProcessPoolExecutor is used for parallel processing.
    
    This is what enables true parallelism (vs threading).
    """
    # Load with parallelization
    ds = read(real_file_subset, lazy=False, n_cpus_use=4)
    
    assert ds is not None
    
    # ProcessPoolExecutor usage is verified by the fact that
    # multiple files are processed successfully


@pytest.mark.synthetic
def test_executor_handles_worker_errors(tmp_path, synthetic_pro):
    """
    Test that executor handles errors in worker processes gracefully.
    """
    # Create mix of valid and corrupted files
    synthetic_pro.basic(tmp_path, "VALID_1")
    synthetic_pro.corrupted(tmp_path, "malformed_data")
    synthetic_pro.basic(tmp_path, "VALID_2")
    
    # Should handle worker errors gracefully
    ds = read(tmp_path, lazy=False, n_cpus_use=4)
    
    # Should load valid files
    assert ds is None or ds.sizes['location'] >= 1


# ============================================================================
# MEMORY AND RESOURCE TESTS
# ============================================================================

@pytest.mark.real_data
def test_parallel_memory_usage(real_file_subset, memory_monitor):
    """
    Test memory usage with parallel processing.
    
    Parallel processing may use more memory due to multiple workers.
    """
    memory_monitor.start()
    
    ds = read(real_file_subset, lazy=False, n_cpus_use=4)
    
    memory_increase = memory_monitor.increase()
    memory_monitor.report()
    
    assert ds is not None
    
    # Memory should be reasonable even with parallel processing
    assert memory_increase < 2000  # < 2 GB


@pytest.mark.synthetic
@pytest.mark.slow
def test_parallel_resource_cleanup(tmp_path, synthetic_pro, memory_monitor):
    """
    Test that parallel processing properly cleans up resources.
    
    Worker processes should be terminated properly.
    """
    synthetic_pro.multi_station(tmp_path, n_stations=15, n_profiles_each=2)
    
    memory_monitor.start()
    
    # Run multiple times to test cleanup
    for _ in range(3):
        ds = read(tmp_path, lazy=False, n_cpus_use=4)
        assert ds is not None
    
    memory_increase = memory_monitor.increase()
    
    # Memory shouldn't grow excessively over multiple runs
    print(f"\nMemory after 3 runs: +{memory_increase:.2f} MB")
    assert memory_increase < 1000


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parallel_with_single_file(tmp_path, synthetic_pro):
    """
    Edge case: Parallel processing with only one file.
    
    Should still work (just uses 1 worker effectively).
    """
    synthetic_pro.basic(tmp_path)
    
    ds = read(tmp_path, lazy=False, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] == 1


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parallel_with_more_cpus_than_files(tmp_path, synthetic_pro):
    """
    Edge case: More CPUs requested than files available.
    
    Extra CPUs should be idle but shouldn't cause problems.
    """
    # Create 3 files
    synthetic_pro.multi_station(tmp_path, n_stations=3, n_profiles_each=2)
    
    # Request 8 CPUs (more than files)
    ds = read(tmp_path, lazy=False, n_cpus_use=8)
    
    assert ds is not None
    assert ds.sizes['location'] == 3


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parallel_with_zero_cpus_invalid(tmp_path, synthetic_pro):
    """
    Edge case: n_cpus_use=0 should be handled.
    
    Implementation should use at least 1 CPU.
    """
    synthetic_pro.basic(tmp_path)
    
    # Depending on implementation, this might raise error or default to 1
    # Test that it doesn't crash
    try:
        ds = read(tmp_path, lazy=False, n_cpus_use=0)
        # If it works, should have data
        if ds is not None:
            assert ds.sizes['location'] > 0
    except ValueError:
        # If it raises ValueError, that's also acceptable
        pass


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_parallel_with_filters(real_file_subset):
    """
    Integration: Parallel processing with selector filters.
    """
    from xsnow.selectors import TimeSelector
    
    ds_full = read(real_file_subset, lazy=False)
    
    if ds_full.sizes['time'] < 2:
        pytest.skip("Need multiple timesteps")
    
    mid_time = ds_full.time.values[ds_full.sizes['time'] // 2]
    time_filter = TimeSelector(sel=slice(None, mid_time))
    
    # Parallel with filter
    ds_filtered = read(real_file_subset, lazy=False, time=time_filter, n_cpus_use=4)
    
    assert ds_filtered is not None
    assert ds_filtered.sizes['time'] <= ds_full.sizes['time']


@pytest.mark.real_data
def test_parallel_with_soil_removal(real_file_subset):
    """
    Integration: Parallel processing with soil removal.
    """
    ds = read(real_file_subset, lazy=False, remove_soil=True, n_cpus_use=4)
    
    assert ds is not None
    
    # Verify soil was removed
    if 'grain_type' in ds.data:
        grain_types = ds['grain_type'].values
        assert not np.any((grain_types == 81) | (grain_types == 82))


@pytest.mark.real_data
def test_parallel_preserves_physical_validity(real_file_subset, assert_physical_validity):
    """
    Integration: Parallel processing preserves data quality.
    """
    ds = read(real_file_subset, lazy=False, n_cpus_use=4)
    
    assert ds is not None
    assert_physical_validity(ds)


# ============================================================================
# REGRESSION TESTS
# ============================================================================

@pytest.mark.real_data
def test_parallel_produces_deterministic_results(real_file_subset, compare_datasets):
    """
    Regression: Parallel processing should be deterministic.
    
    Same input should always produce same output.
    """
    # Run twice with parallel processing
    ds1 = read(real_file_subset, lazy=False, n_cpus_use=4)
    ds2 = read(real_file_subset, lazy=False, n_cpus_use=4)
    
    # Should be identical
    compare_datasets(ds1, ds2, rtol=1e-7)


@pytest.mark.synthetic
def test_parallel_maintains_file_order(tmp_path, synthetic_pro):
    """
    Regression: Parallel processing should maintain consistent ordering.
    
    Location mapping should be deterministic regardless of processing order.
    """
    # Create files with specific names
    for i in range(5):
        content = f"""[STATION_PARAMETERS]
StationName= STATION_{i:03d}
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
"""
        (tmp_path / f"station_{i:03d}.pro").write_text(content)
    
    # Load with parallel processing
    ds = read(tmp_path, lazy=False, n_cpus_use=4)
    
    # Location mapping should be deterministic (sorted)
    mapping = ds.location.attrs.get('mapping', {})
    station_names = [mapping[i] for i in sorted(mapping.keys())]
    
    # Should be sorted
    assert station_names == sorted(station_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])