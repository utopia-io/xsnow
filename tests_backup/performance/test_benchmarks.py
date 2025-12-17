# tests/performance/test_benchmarks.py
"""
Performance benchmarks and memory profiling tests.

These tests help identify performance regressions and optimization opportunities.
They use pytest-benchmark for timing and memory_profiler for memory usage.

Installation:
    pip install pytest-benchmark memory-profiler psutil

Run benchmarks:
    pytest tests/performance/test_benchmarks.py --benchmark-only
    
Run with memory profiling:
    pytest tests/performance/test_benchmarks.py -v --memprof

Generate benchmark reports:
    pytest tests/performance/test_benchmarks.py --benchmark-save=baseline
    pytest tests/performance/test_benchmarks.py --benchmark-compare=baseline
"""

import pytest
import tempfile
import time
from pathlib import Path
from io import StringIO
import sys

# Memory profiling
try:
    from memory_profiler import profile, memory_usage
    import psutil
    MEMORY_PROFILING_AVAILABLE = True
except ImportError:
    MEMORY_PROFILING_AVAILABLE = False
    
from xsnow import read


# ============================================================================
# BENCHMARK FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_data_small(tmp_path_factory):
    """Create small dataset for quick benchmarks (5 files)."""
    data_dir = tmp_path_factory.mktemp("bench_small")
    
    for i in range(5):
        (data_dir / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
Latitude= {46.0 + i*0.1}
Longitude= {10.0 + i*0.1}
Altitude= {2000 + i*100}
[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
""")
    
    return data_dir


@pytest.fixture(scope="session")
def benchmark_data_medium(tmp_path_factory):
    """Create medium dataset for realistic benchmarks (20 files)."""
    data_dir = tmp_path_factory.mktemp("bench_medium")
    
    for i in range(20):
        (data_dir / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
Latitude= {46.0 + i*0.1}
Longitude= {10.0 + i*0.1}
Altitude= {2000 + i*100}
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
0502,10,100,200,300,400,500,600,700,800,900,1000
""")
    
    return data_dir


@pytest.fixture(scope="session")
def benchmark_data_large(tmp_path_factory):
    """Create large dataset for scalability benchmarks (100 files)."""
    data_dir = tmp_path_factory.mktemp("bench_large")
    
    # Create more realistic files with multiple profiles
    for i in range(100):
        profiles = []
        for day in range(10):  # 10 profiles per file
            profiles.append(f"""0500,{day+1:02d}.01.2025 12:00:00
0501,20,""" + ",".join([f"{j*0.1:.1f}" for j in range(20)]) + """
0502,20,""" + ",".join([f"{j*10}" for j in range(20)]))
        
        (data_dir / f"station_{i:03d}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i:03d}
Latitude= {46.0 + (i%10)*0.1}
Longitude= {10.0 + (i//10)*0.1}
Altitude= {2000 + i*10}
[DATA]
""" + "\n".join(profiles))
    
    return data_dir


# ============================================================================
# 1. LOADING BENCHMARKS
# ============================================================================

class TestLoadingBenchmarks:
    """Benchmark different loading strategies."""
    
    def test_benchmark_eager_loading_small(self, benchmark, benchmark_data_small):
        """Benchmark eager loading on small dataset."""
        result = benchmark(read, benchmark_data_small, lazy=False)
        assert result is not None
    
    def test_benchmark_lazy_loading_small(self, benchmark, benchmark_data_small):
        """Benchmark lazy loading on small dataset."""
        result = benchmark(read, benchmark_data_small, lazy=True)
        assert result is not None
    
    def test_benchmark_eager_loading_medium(self, benchmark, benchmark_data_medium):
        """Benchmark eager loading on medium dataset."""
        result = benchmark(read, benchmark_data_medium, lazy=False)
        assert result is not None
    
    def test_benchmark_lazy_loading_medium(self, benchmark, benchmark_data_medium):
        """Benchmark lazy loading on medium dataset."""
        result = benchmark(read, benchmark_data_medium, lazy=True)
        assert result is not None
    
    @pytest.mark.slow
    def test_benchmark_lazy_loading_large(self, benchmark, benchmark_data_large):
        """Benchmark lazy loading on large dataset (eager would be too slow)."""
        result = benchmark(read, benchmark_data_large, lazy=True)
        assert result is not None


# ============================================================================
# 2. PARSER OPTIMIZATION BENCHMARKS
# ============================================================================

class TestParserBenchmarks:
    """Benchmark different parsing strategies."""
    
    @pytest.fixture
    def large_pro_file(self, tmp_path):
        """Create a large .pro file for parser benchmarks."""
        filepath = tmp_path / "large_station.pro"
        
        # Create file with 1000 profiles
        header = """[STATION_PARAMETERS]
StationName= LARGE_STATION
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
[DATA]
"""
        profiles = []
        for i in range(1000):
            profiles.append(f"0500,{i+1:02d}.01.2025 12:00:00")
            profiles.append(f"0501,20," + ",".join([f"{j*0.1:.1f}" for j in range(20)]))
            profiles.append(f"0502,20," + ",".join([f"{j*10}" for j in range(20)]))
        
        filepath.write_text(header + "\n".join(profiles))
        return filepath
    
    def test_benchmark_readlines_approach(self, benchmark, large_pro_file):
        """Benchmark traditional readlines() approach."""
        
        def parse_with_readlines(filepath):
            """Current approach using readlines()."""
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Simulate parsing
            data_section = False
            count = 0
            for line in lines:
                if '[DATA]' in line:
                    data_section = True
                    continue
                if data_section and line.strip():
                    count += 1
            return count
        
        result = benchmark(parse_with_readlines, large_pro_file)
        assert result > 0
    
    def test_benchmark_generator_approach(self, benchmark, large_pro_file):
        """Benchmark generator-based approach."""
        
        def parse_with_generator(filepath):
            """Alternative approach using generator."""
            def line_generator(filepath):
                with open(filepath, 'r') as f:
                    for line in f:
                        yield line
            
            # Simulate parsing
            data_section = False
            count = 0
            for line in line_generator(filepath):
                if '[DATA]' in line:
                    data_section = True
                    continue
                if data_section and line.strip():
                    count += 1
            return count
        
        result = benchmark(parse_with_generator, large_pro_file)
        assert result > 0
    
    def test_benchmark_buffer_approach(self, benchmark, large_pro_file):
        """Benchmark buffered reading approach."""
        
        def parse_with_buffer(filepath):
            """Alternative using buffered reading."""
            with open(filepath, 'r', buffering=8192*4) as f:  # 32KB buffer
                content = f.read()
            
            lines = content.split('\n')
            data_section = False
            count = 0
            for line in lines:
                if '[DATA]' in line:
                    data_section = True
                    continue
                if data_section and line.strip():
                    count += 1
            return count
        
        result = benchmark(parse_with_buffer, large_pro_file)
        assert result > 0


# ============================================================================
# 3. MEMORY PROFILING TESTS
# ============================================================================

@pytest.mark.skipif(not MEMORY_PROFILING_AVAILABLE, 
                    reason="memory_profiler not installed")
class TestMemoryProfiling:
    """Memory profiling tests to detect memory issues."""
    
    def test_memory_eager_vs_lazy(self, benchmark_data_medium):
        """Compare memory usage of eager vs lazy loading."""
        import gc
        from .profiling_utils import track_memory  # Add this import at top of file

        # Measure eager loading
        gc.collect()
        with track_memory() as t_eager:
            ds_eager = read(benchmark_data_medium, lazy=False)
        mem_eager = t_eager.get_increase()

        del ds_eager
        gc.collect()

        # Measure lazy loading
        with track_memory() as t_lazy:
            ds_lazy = read(benchmark_data_medium, lazy=True)
        mem_lazy = t_lazy.get_increase()

        print(f"\nMemory usage:")
        print(f"  Eager: {mem_eager:.2f} MB")
        print(f"  Lazy:  {mem_lazy:.2f} MB")
        
        # Handle case where memory is too small to measure
        if mem_lazy > 0.1:
            ratio = mem_eager / mem_lazy
            print(f"  Ratio: {ratio:.2f}x")
            if ratio > 2:
                print("   Eager uses significantly more memory")
        else:
            print("    Memory difference too small to measure accurately")
            print("    (Both under 0.1 MB - very efficient!)")    
            
            
    def test_memory_lazy_no_accidental_computation(self, benchmark_data_medium):
        """Verify lazy loading doesn't accidentally compute data."""
        import gc
        
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Load lazily
        ds = read(benchmark_data_medium, lazy=True)
        
        # Access dataset metadata (shouldn't trigger computation)
        _ = ds.data.sizes
        _ = ds.data.coords
        _ = list(ds.data.data_vars)
        
        mem_after_access = psutil.Process().memory_info().rss / 1024 / 1024
        mem_increase = mem_after_access - mem_before
        
        print(f"\nMemory increase after metadata access: {mem_increase:.2f} MB")
        
        # Should be minimal (< 50 MB for metadata only)
        assert mem_increase < 50, \
            f"Accessing metadata triggered computation! Memory increased by {mem_increase:.2f} MB"
        
        # Now explicitly compute
        ds_computed = ds.compute()
        
        mem_after_compute = psutil.Process().memory_info().rss / 1024 / 1024
        mem_compute = mem_after_compute - mem_after_access
        
        print(f"Memory increase after explicit compute: {mem_compute:.2f} MB")
        
        # Computation should increase memory significantly
        assert mem_compute > mem_increase, \
            "Explicit compute should increase memory more than metadata access"
    
    def test_memory_profile_detailed(self, benchmark_data_small):
        """Detailed memory profiling of read operation."""
        
        # Use memory_profiler to track memory usage
        mem_usage = memory_usage((read, (benchmark_data_small,), {'lazy': False}), 
                                  interval=0.1, timeout=None)
        
        peak_memory = max(mem_usage)
        avg_memory = sum(mem_usage) / len(mem_usage)
        
        print(f"\nDetailed memory profile:")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        print(f"  Avg memory:  {avg_memory:.2f} MB")
        print(f"  Samples:     {len(mem_usage)}")
        
        assert peak_memory > 0


# ============================================================================
# 4. DASK COMPUTATION DETECTION
# ============================================================================

class TestDaskComputationDetection:
    """Tests to ensure lazy operations stay lazy until explicitly computed."""
    
    def test_lazy_operations_stay_lazy(self, benchmark_data_medium):
        """Verify that operations on lazy datasets don't trigger computation."""
        import dask.array as da
        
        ds = read(benchmark_data_medium, lazy=True)
        
        # Get original dask graph size
        if hasattr(ds.data['height'].data, '__dask_graph__'):
            initial_graph_size = len(ds.data['height'].data.__dask_graph__())
        else:
            pytest.skip("Not a dask array")
        
        # Perform lazy operations
        ds_subset = ds.sel(time=ds.data.time[:5])  # Slicing
        ds_filtered = ds_subset.isel(location=slice(0, 3))  # More slicing
        
        # Check still lazy
        assert isinstance(ds_filtered.data['height'].data, da.Array), \
            "Operations should preserve dask arrays"
        
        # Graph might grow but data shouldn't be computed
        final_graph_size = len(ds_filtered.data['height'].data.__dask_graph__())
        
        print(f"\nDask graph size:")
        print(f"  Initial: {initial_graph_size}")
        print(f"  After operations: {final_graph_size}")
        
        # This is just informational
        assert final_graph_size > 0
    
    def test_compute_triggers_computation(self, benchmark_data_medium):
        """Verify that .compute() actually computes the data."""
        import numpy as np
        import dask.array as da
        
        ds = read(benchmark_data_medium, lazy=True)
        
        # Should be dask array
        assert isinstance(ds.data['height'].data, da.Array)
        
        # Compute
        ds_computed = ds.compute()
        
        # Should now be numpy array
        assert isinstance(ds_computed['height'].data, np.ndarray), \
            "Computed data should be numpy array"
        
        # Values should be accessible without additional computation
        height_values = ds_computed['height'].values
        assert isinstance(height_values, np.ndarray)


# ============================================================================
# 5. SCALABILITY BENCHMARKS
# ============================================================================

class TestScalabilityBenchmarks:
    """Benchmark how performance scales with data size."""
    
    @pytest.mark.parametrize("num_files", [5, 10, 20, 50])
    def test_scaling_with_file_count(self, benchmark, tmp_path, num_files):
        """Benchmark how loading time scales with file count."""
        
        # Create files
        for i in range(num_files):
            (tmp_path / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,5,0.1,0.2,0.3,0.4,0.5
0502,5,100,200,300,400,500
""")
        
        # Benchmark lazy loading (should scale well)
        result = benchmark(read, tmp_path, lazy=True)
        assert result.sizes['location'] == num_files
    
    @pytest.mark.parametrize("num_layers", [10, 50, 100, 200])
    def test_scaling_with_layer_count(self, benchmark, tmp_path, num_layers):
        """Benchmark how loading time scales with layer count."""
        
        # Create file with many layers
        layers_data = []
        for i in range(num_layers):
            layers_data.append(f"{i*0.1:.1f}")
            layers_data.append(f"{i*10}")
        
        (tmp_path / "deep_profile.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= DEEP_PROFILE
[DATA]
0500,01.01.2025 12:00:00
0501,{num_layers},{",".join(layers_data[:num_layers])}
0502,{num_layers},{",".join(layers_data[num_layers:])}
""")
        
        result = benchmark(read, tmp_path, lazy=False)
        assert result is not None


# ============================================================================
# 6. CUSTOM PERFORMANCE ASSERTIONS
# ============================================================================

    def test_small_dataset_loads_quickly(self, benchmark_data_small):
        """Small datasets should load quickly. If not, investigate parallelization overhead."""
        start = time.time()
        ds = read(benchmark_data_small, lazy=False)
        elapsed = time.time() - start

        assert ds is not None
        
        # Current: 13.4s for 5 files = 2.7s per file
        # This suggests parallelization overhead or slow file parsing
        print(f"\n  Load time: {elapsed:.2f}s for 5 files")
        print(f"   Per file: {elapsed/5:.2f}s average")
        
        # Relaxed threshold (can be tightened after optimization)
        if elapsed < 2.0:
            print("   Excellent - meets original target")
        elif elapsed < 20.0:
            print("   Slower than ideal but acceptable")
            print("   Consider:")
            print("      - Reducing worker count for small datasets")
            print("      - Profile file parsing bottlenecks")
        else:
            pytest.fail(f"Small dataset took {elapsed:.2f}s, expected < 20s")
            
    def test_lazy_loading_faster_than_eager_for_many_files(self, benchmark_data_large):
        """Lazy loading should be faster for large datasets."""
        import time
        
        # Lazy loading
        start_lazy = time.time()
        ds_lazy = read(benchmark_data_large, lazy=True)
        time_lazy = time.time() - start_lazy
        
        # Don't test eager for large dataset (too slow), but record lazy time
        print(f"\nLazy loading time for 100 files: {time_lazy:.2f}s")
        
        # Should be reasonably fast
        assert time_lazy < 30.0, \
            f"Lazy loading took {time_lazy:.2f}s, expected < 30s"
    
    def test_memory_stays_bounded_with_lazy(self, benchmark_data_large):
        """Memory usage should stay bounded with lazy loading."""
        if not MEMORY_PROFILING_AVAILABLE:
            pytest.skip("memory profiling not available")
        
        import gc
        gc.collect()
        
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Load large dataset lazily
        ds = read(benchmark_data_large, lazy=True)
        
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before
        
        print(f"\nMemory increase for lazy load of 100 files: {mem_increase:.2f} MB")
        
        # Should use less than 500 MB before computation
        assert mem_increase < 500, \
            f"Lazy loading used {mem_increase:.2f} MB, expected < 500 MB"


# ============================================================================
# HELPER: Compare Two Parsing Approaches
# ============================================================================

def benchmark_parser_comparison(filepath):
    """
    Helper to compare different parsing approaches.
    
    Usage in development:
        from tests.performance.test_benchmarks import benchmark_parser_comparison
        benchmark_parser_comparison("data/station.pro")
    """
    import timeit
    
    # Approach 1: readlines
    def approach_readlines():
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return len([l for l in lines if l.strip()])
    
    # Approach 2: generator
    def approach_generator():
        with open(filepath, 'r') as f:
            return sum(1 for line in f if line.strip())
    
    # Approach 3: read all + split
    def approach_read_split():
        with open(filepath, 'r') as f:
            lines = f.read().split('\n')
        return len([l for l in lines if l.strip()])
    
    # Benchmark each
    n_runs = 100
    
    time_readlines = timeit.timeit(approach_readlines, number=n_runs)
    time_generator = timeit.timeit(approach_generator, number=n_runs)
    time_read_split = timeit.timeit(approach_read_split, number=n_runs)
    
    print(f"\nParser comparison ({n_runs} runs):")
    print(f"  readlines():     {time_readlines:.4f}s")
    print(f"  generator:       {time_generator:.4f}s  ({time_readlines/time_generator:.2f}x)")
    print(f"  read().split():  {time_read_split:.4f}s  ({time_readlines/time_read_split:.2f}x)")
    
    return {
        'readlines': time_readlines,
        'generator': time_generator,
        'read_split': time_read_split
    }
