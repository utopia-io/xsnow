# tests/performance/test_parser_optimization_improved.py
"""
Realistic parser optimization tests using multiple files (like actual xsnow usage).

This tests:
1. Small dataset: 5 files × 20 profiles = 100 profiles total
2. Large dataset: 20 files × 100 profiles = 2,000 profiles total  
3. Huge dataset: 50 files × 200 profiles = 10,000 profiles total

Run:
    pytest tests/performance/test_parser_optimization_improved.py -v -s
"""

import pytest
from pathlib import Path
from .profiling_utils import Benchmark, track_memory
import gc


# ============================================================================
# PARSER IMPLEMENTATIONS
# ============================================================================

def parse_pro_readlines(filepath: Path) -> dict:
    """Current approach: Read all lines into memory at once."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = {'station_name': None, 'profiles': [], 'header': {}}
    in_data_section = False
    current_profile = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('[STATION_PARAMETERS]'):
            continue
        elif line.startswith('[DATA]'):
            in_data_section = True
            continue
        elif line.startswith('StationName='):
            data['station_name'] = line.split('=', 1)[1].strip()
        elif in_data_section and line:
            if line.startswith('0500'):
                if current_profile:
                    data['profiles'].append(current_profile)
                current_profile = [line]
            else:
                current_profile.append(line)
    
    if current_profile:
        data['profiles'].append(current_profile)
    return data


def parse_pro_generator(filepath: Path) -> dict:
    """Alternative: Use generator for memory efficiency."""
    def line_generator(fp):
        with open(fp, 'r') as f:
            for line in f:
                yield line.strip()
    
    data = {'station_name': None, 'profiles': [], 'header': {}}
    in_data_section = False
    current_profile = []
    
    for line in line_generator(filepath):
        if line.startswith('[STATION_PARAMETERS]'):
            continue
        elif line.startswith('[DATA]'):
            in_data_section = True
            continue
        elif line.startswith('StationName='):
            data['station_name'] = line.split('=', 1)[1].strip()
        elif in_data_section and line:
            if line.startswith('0500'):
                if current_profile:
                    data['profiles'].append(current_profile)
                current_profile = [line]
            else:
                current_profile.append(line)
    
    if current_profile:
        data['profiles'].append(current_profile)
    return data


def parse_directory_readlines(directory: Path) -> list:
    """Parse all .pro files in directory using readlines."""
    results = []
    for filepath in sorted(directory.glob("*.pro")):
        results.append(parse_pro_readlines(filepath))
    return results


def parse_directory_generator(directory: Path) -> list:
    """Parse all .pro files in directory using generator."""
    results = []
    for filepath in sorted(directory.glob("*.pro")):
        results.append(parse_pro_generator(filepath))
    return results


# ============================================================================
# TEST FIXTURES - Multiple Files per Dataset
# ============================================================================

@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    """
    Small dataset: 5 files × 20 profiles each = 100 profiles total
    Typical use case: Small study area or short time period
    """
    data_dir = tmp_path_factory.mktemp("small_dataset")
    
    for file_idx in range(5):
        profiles = []
        for profile_idx in range(20):
            profiles.extend([
                f"0500,{profile_idx+1:02d}.01.2025 {file_idx:02d}:00:00",
                f"0501,10," + ",".join([f"{i*0.1:.1f}" for i in range(10)]),
                f"0502,10," + ",".join([f"{i*100}" for i in range(10)])
            ])
        
        (data_dir / f"station_{file_idx:03d}.pro").write_text(
            f"[STATION_PARAMETERS]\nStationName=STATION_{file_idx:03d}\n\n[DATA]\n" + 
            "\n".join(profiles)
        )
    
    return data_dir


@pytest.fixture(scope="session")
def large_dataset(tmp_path_factory):
    """
    Large dataset: 20 files × 100 profiles each = 2,000 profiles total
    Typical use case: Regional study, full season
    """
    data_dir = tmp_path_factory.mktemp("large_dataset")
    
    for file_idx in range(20):
        profiles = []
        for profile_idx in range(100):
            profiles.extend([
                f"0500,{profile_idx+1:02d}.01.2025 12:00:00",
                f"0501,20," + ",".join([f"{i*0.1:.1f}" for i in range(20)]),
                f"0502,20," + ",".join([f"{i*10}" for i in range(20)])
            ])
        
        (data_dir / f"station_{file_idx:03d}.pro").write_text(
            f"[STATION_PARAMETERS]\nStationName=STATION_{file_idx:03d}\n\n[DATA]\n" + 
            "\n".join(profiles)
        )
    
    return data_dir


@pytest.fixture(scope="session")
def huge_dataset(tmp_path_factory):
    """
    Huge dataset: 50 files × 200 profiles each = 10,000 profiles total
    Stress test: Large operational forecast system
    """
    data_dir = tmp_path_factory.mktemp("huge_dataset")
    
    for file_idx in range(50):
        profiles = []
        for profile_idx in range(200):
            profiles.extend([
                f"0500,{profile_idx+1:02d}.01.2025 12:00:00",
                f"0501,30," + ",".join([f"{i*0.1:.1f}" for i in range(30)]),
                f"0502,30," + ",".join([f"{i*10}" for i in range(30)])
            ])
        
        (data_dir / f"station_{file_idx:03d}.pro").write_text(
            f"[STATION_PARAMETERS]\nStationName=STATION_{file_idx:03d}\n\n[DATA]\n" + 
            "\n".join(profiles)
        )
    
    return data_dir


# ============================================================================
# ORGANIZED TIMING TESTS
# ============================================================================

class TestTimingByDatasetSize:
    """Compare timing for different dataset sizes."""
    
    def test_timing_small_dataset(self, small_dataset):
        """
        SMALL DATASET: 5 files, 100 profiles total
        Expected: Both approaches should be very fast, negligible difference
        """
        bench = Benchmark()
        
        bench.run("readlines", parse_directory_readlines, small_dataset, n_runs=50)
        bench.run("generator", parse_directory_generator, small_dataset, n_runs=50)
        
        print("\n" + "="*70)
        print("TIMING: SMALL DATASET (5 files × 20 profiles = 100 total)")
        print("="*70)
        bench.compare("readlines")
        print("="*70)
    
    def test_timing_large_dataset(self, large_dataset):
        """
        LARGE DATASET: 20 files, 2,000 profiles total
        Expected: Differences may start to appear here
        """
        bench = Benchmark()
        
        bench.run("readlines", parse_directory_readlines, large_dataset, n_runs=20)
        bench.run("generator", parse_directory_generator, large_dataset, n_runs=20)
        
        print("\n" + "="*70)
        print("TIMING: LARGE DATASET (20 files × 100 profiles = 2,000 total)")
        print("="*70)
        bench.compare("readlines")
        print("="*70)
    
    @pytest.mark.slow
    def test_timing_huge_dataset(self, huge_dataset):
        """
        HUGE DATASET: 50 files, 10,000 profiles total
        Expected: This is where generator should shine (if it does)
        """
        bench = Benchmark()
        
        bench.run("readlines", parse_directory_readlines, huge_dataset, n_runs=10)
        bench.run("generator", parse_directory_generator, huge_dataset, n_runs=10)
        
        print("\n" + "="*70)
        print("TIMING: HUGE DATASET (50 files × 200 profiles = 10,000 total)")
        print("="*70)
        bench.compare("readlines")
        print("="*70)


# ============================================================================
# ORGANIZED MEMORY TESTS
# ============================================================================

class TestMemoryByDatasetSize:
    """Compare memory usage for different dataset sizes."""
    
    def test_memory_small_dataset(self, small_dataset):
        """Memory test: Small dataset."""
        gc.collect()
        
        with track_memory("readlines") as t1:
            results1 = parse_directory_readlines(small_dataset)
        mem1 = t1.get_increase()
        
        gc.collect()
        
        with track_memory("generator") as t2:
            results2 = parse_directory_generator(small_dataset)
        mem2 = t2.get_increase()
        
        print("\n" + "="*70)
        print("MEMORY: SMALL DATASET (5 files × 20 profiles = 100 total)")
        print("="*70)
        print(f"readlines: {mem1:6.2f} MB")
        print(f"generator: {mem2:6.2f} MB")
        
        if mem1 > mem2 and mem2 > 0.01:
            savings = mem1 - mem2
            ratio = mem1 / mem2
            print(f"Savings:   {savings:6.2f} MB ({ratio:.2f}x)")
        elif mem2 > mem1 and mem1 > 0.01:
            overhead = mem2 - mem1
            ratio = mem2 / mem1
            print(f"Overhead:  {overhead:6.2f} MB ({ratio:.2f}x)")
        else:
            print(f"Difference: Too small to measure accurately")
        print("="*70)
        
        assert len(results1) == len(results2) == 5
    
    def test_memory_large_dataset(self, large_dataset):
        """Memory test: Large dataset."""
        gc.collect()
        
        with track_memory("readlines") as t1:
            results1 = parse_directory_readlines(large_dataset)
        mem1 = t1.get_increase()
        
        gc.collect()
        
        with track_memory("generator") as t2:
            results2 = parse_directory_generator(large_dataset)
        mem2 = t2.get_increase()
        
        print("\n" + "="*70)
        print("MEMORY: LARGE DATASET (20 files × 100 profiles = 2,000 total)")
        print("="*70)
        print(f"readlines: {mem1:6.2f} MB")
        print(f"generator: {mem2:6.2f} MB")
        
        if mem1 > mem2 and mem2 > 0.1:
            savings = mem1 - mem2
            ratio = mem1 / mem2
            print(f"Savings:   {savings:6.2f} MB ({ratio:.2f}x)")
        elif mem2 > mem1 and mem1 > 0.1:
            overhead = mem2 - mem1
            ratio = mem2 / mem1
            print(f"Overhead:  {overhead:6.2f} MB ({ratio:.2f}x)")
        else:
            print(f"Difference: {abs(mem1-mem2):.2f} MB (minimal)")
        print("="*70)
        
        assert len(results1) == len(results2) == 20
    
    @pytest.mark.slow
    def test_memory_huge_dataset(self, huge_dataset):
        """Memory test: Huge dataset - where differences should appear."""
        gc.collect()
        
        with track_memory("readlines") as t1:
            results1 = parse_directory_readlines(huge_dataset)
        mem1 = t1.get_increase()
        
        gc.collect()
        
        with track_memory("generator") as t2:
            results2 = parse_directory_generator(huge_dataset)
        mem2 = t2.get_increase()
        
        print("\n" + "="*70)
        print("MEMORY: HUGE DATASET (50 files × 200 profiles = 10,000 total)")
        print("="*70)
        print(f"readlines: {mem1:6.2f} MB")
        print(f"generator: {mem2:6.2f} MB")
        
        if mem1 > mem2 and mem2 > 0.1:
            savings = mem1 - mem2
            ratio = mem1 / mem2
            print(f"Savings:   {savings:6.2f} MB ({ratio:.2f}x)")
            print(f"Reduction: {(savings/mem1)*100:.1f}%")
        elif mem2 > mem1 and mem1 > 0.1:
            overhead = mem2 - mem1
            ratio = mem2 / mem1
            print(f"Overhead:  {overhead:6.2f} MB ({ratio:.2f}x)")
        else:
            print(f"Difference: {abs(mem1-mem2):.2f} MB")
        print("="*70)
        
        assert len(results1) == len(results2) == 50


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

class TestComprehensiveComparison:
    """Run all scenarios and provide final recommendation."""
    
    def test_comprehensive_comparison(self, small_dataset, large_dataset, huge_dataset):
        """
        Compare all dataset sizes and provide comprehensive recommendation.
        
        This test runs everything and gives you a clear answer.
        """
        results = {}
        
        # Small dataset
        print("\n" + "="*70)
        print("COMPREHENSIVE PARSER COMPARISON")
        print("="*70)
        print("\n Testing SMALL dataset (5 files, 100 profiles)...")
        
        bench_small = Benchmark()
        bench_small.run("readlines", parse_directory_readlines, small_dataset, n_runs=30)
        bench_small.run("generator", parse_directory_generator, small_dataset, n_runs=30)
        
        results['small'] = {
            'readlines_time': bench_small.results['readlines']['mean'],
            'generator_time': bench_small.results['generator']['mean'],
            'files': 5,
            'profiles': 100
        }
        
        # Large dataset
        print("\n Testing LARGE dataset (20 files, 2,000 profiles)...")
        
        bench_large = Benchmark()
        bench_large.run("readlines", parse_directory_readlines, large_dataset, n_runs=20)
        bench_large.run("generator", parse_directory_generator, large_dataset, n_runs=20)
        
        results['large'] = {
            'readlines_time': bench_large.results['readlines']['mean'],
            'generator_time': bench_large.results['generator']['mean'],
            'files': 20,
            'profiles': 2000
        }
        
        # Huge dataset
        print("\n Testing HUGE dataset (50 files, 10,000 profiles)...")
        
        bench_huge = Benchmark()
        bench_huge.run("readlines", parse_directory_readlines, huge_dataset, n_runs=10)
        bench_huge.run("generator", parse_directory_generator, huge_dataset, n_runs=10)
        
        results['huge'] = {
            'readlines_time': bench_huge.results['readlines']['mean'],
            'generator_time': bench_huge.results['generator']['mean'],
            'files': 50,
            'profiles': 10000
        }
        
        # Memory comparison (large dataset only for speed)
        gc.collect()
        with track_memory() as t1:
            parse_directory_readlines(large_dataset)
        mem_readlines = t1.get_increase()
        
        gc.collect()
        with track_memory() as t2:
            parse_directory_generator(large_dataset)
        mem_generator = t2.get_increase()
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        
        print("\n TIMING RESULTS:")
        print("-" * 70)
        print(f"{'Dataset':<15} {'Files':<8} {'Profiles':<10} {'readlines':<12} {'generator':<12} {'Winner'}")
        print("-" * 70)
        
        for size_name, data in results.items():
            r_time = data['readlines_time'] * 1000  # Convert to ms
            g_time = data['generator_time'] * 1000
            winner = "generator " if g_time < r_time * 0.95 else "readlines " if r_time < g_time * 0.95 else "tie"
            
            print(f"{size_name.upper():<15} {data['files']:<8} {data['profiles']:<10} "
                  f"{r_time:>10.2f}ms  {g_time:>10.2f}ms  {winner}")
        
        print("-" * 70)
        
        print("\n MEMORY RESULTS (Large dataset):")
        print("-" * 70)
        print(f"readlines: {mem_readlines:6.2f} MB")
        print(f"generator: {mem_generator:6.2f} MB")
        
        if abs(mem_readlines - mem_generator) > 0.5:
            if mem_generator < mem_readlines:
                savings = mem_readlines - mem_generator
                print(f"Savings:   {savings:6.2f} MB ({(savings/mem_readlines)*100:.1f}% reduction)")
            else:
                overhead = mem_generator - mem_readlines
                print(f"Overhead:  {overhead:6.2f} MB ({(overhead/mem_readlines)*100:.1f}% increase)")
        else:
            print(f"Difference: Negligible ({abs(mem_readlines - mem_generator):.2f} MB)")
        
        print("-" * 70)
        
        # Recommendation logic
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        # Calculate average speedup across all sizes
        speedups = []
        for data in results.values():
            speedup = data['readlines_time'] / data['generator_time']
            speedups.append(speedup)
        avg_speedup = sum(speedups) / len(speedups)
        
        mem_savings = mem_readlines - mem_generator
        significant_mem_savings = abs(mem_savings) > 5  # >5MB is significant
        
        if avg_speedup > 1.1 and mem_savings > 0:
            print("\nUSE GENERATOR")
            print(f"   Reason: Faster ({avg_speedup:.2f}x) and uses less memory")
        elif avg_speedup > 1.2:
            print("\nUSE GENERATOR")
            print(f"   Reason: Significantly faster ({avg_speedup:.2f}x)")
        elif significant_mem_savings and mem_savings > 0:
            print("\nUSE GENERATOR")
            print(f"   Reason: Saves {mem_savings:.1f} MB of memory")
        elif avg_speedup < 0.9 or (significant_mem_savings and mem_savings < 0):
            print("\nKEEP READLINES")
            print(f"   Reason: Simpler code, generator shows no clear benefit")
        else:
            print("\nEITHER APPROACH IS FINE")
            print(f"   Reason: Performance difference is negligible")
            print(f"   - Speed: {avg_speedup:.2f}x (< 10% difference)")
            print(f"   - Memory: {mem_savings:.1f} MB difference (< 5 MB)")
            print(f"   - Recommendation: Use readlines() for code simplicity")
        
        print("\n" + "="*70)


# ============================================================================
# PYTEST-BENCHMARK INTEGRATION
# ============================================================================

class TestBenchmarkIntegration:
    """Standard pytest-benchmark integration for CI/CD."""
    
    def test_bench_readlines_small(self, benchmark, small_dataset):
        """Benchmark readlines on small dataset."""
        result = benchmark(parse_directory_readlines, small_dataset)
        assert len(result) == 5
    
    def test_bench_generator_small(self, benchmark, small_dataset):
        """Benchmark generator on small dataset."""
        result = benchmark(parse_directory_generator, small_dataset)
        assert len(result) == 5
    
    def test_bench_readlines_large(self, benchmark, large_dataset):
        """Benchmark readlines on large dataset."""
        result = benchmark(parse_directory_readlines, large_dataset)
        assert len(result) == 20
    
    def test_bench_generator_large(self, benchmark, large_dataset):
        """Benchmark generator on large dataset."""
        result = benchmark(parse_directory_generator, large_dataset)
        assert len(result) == 20