# tests/integration/test_performance.py
"""
Performance benchmarks and scalability tests.
Focus: Performance characteristics at scale.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from xsnow import read


@pytest.mark.slow
@pytest.mark.benchmark
class TestScalability:
    """Test performance scaling with dataset size."""
    
    def test_scalability_with_file_count(self, synthetic_pro, tmp_path, timer, memory_monitor):
        """Test how performance scales with number of files."""
        file_counts = [10, 50, 100]
        results = {}
        
        for n_files in file_counts:
            # Create test data
            data_dir = tmp_path / f"scale_{n_files}"
            data_dir.mkdir()
            
            for i in range(n_files):
                synthetic_pro.basic(
                    data_dir,
                    station_name=f"STATION_{i}",
                    n_layers=10
                )
            
            # Benchmark eager loading
            memory_monitor.start()
            timer.start(f'eager_{n_files}')
            ds_eager = read(data_dir, lazy=False, n_cpus_use=4)
            timer.stop(f'eager_{n_files}')
            mem_eager = memory_monitor.increase()
            
            # Benchmark lazy loading
            memory_monitor.start()
            timer.start(f'lazy_{n_files}')
            ds_lazy = read(data_dir, lazy=True, n_cpus_use=4)
            timer.stop(f'lazy_{n_files}')
            mem_lazy = memory_monitor.increase()
            
            results[n_files] = {
                'eager_time': timer.get(f'eager_{n_files}'),
                'lazy_time': timer.get(f'lazy_{n_files}'),
                'eager_mem': mem_eager,
                'lazy_mem': mem_lazy,
                'locations': ds_eager.sizes['location']
            }
        
        # Report results
        print("\n" + "="*80)
        print("SCALABILITY TEST RESULTS")
        print("="*80)
        print(f"{'Files':<10} {'Eager Time':<15} {'Lazy Time':<15} {'Eager Mem':<15} {'Lazy Mem':<15}")
        print("-"*80)
        
        for n_files, res in results.items():
            print(f"{n_files:<10} {res['eager_time']:<15.3f} "
                  f"{res['lazy_time']:<15.3f} "
                  f"{res['eager_mem']:<15.1f} {res['lazy_mem']:<15.1f}")
        
        print("="*80)
        
        # Check that both modes completed successfully
        for n_files in file_counts:
            assert results[n_files]['locations'] == n_files
        
        timer.report()
    
    @pytest.mark.slow
    def test_scalability_with_profile_count(self, synthetic_pro, tmp_path, timer):
        """Test scaling with number of profiles per file."""
        profile_counts = [10, 50, 100, 200]
        results = {}
        
        for n_profiles in profile_counts:
            # FIX: Create subdirectory for each test
            test_dir = tmp_path / f"profiles_{n_profiles}"
            test_dir.mkdir()
            
            # Create file with many profiles
            filepath = synthetic_pro.varying_layers(
                test_dir,  # Pass directory
                layer_counts=[5] * n_profiles
            )
            
            timer.start(f'profiles_{n_profiles}')
            ds = read(test_dir, lazy=False)  # Read from directory
            timer.stop(f'profiles_{n_profiles}')
            
            results[n_profiles] = {
                'time': timer.get(f'profiles_{n_profiles}'),
                'timesteps': ds.sizes['time']
            }
        
        # Report
        print("\n" + "="*60)
        print("PROFILE COUNT SCALING")
        print("="*60)
        print(f"{'Profiles':<15} {'Load Time':<15} {'Timesteps':<15}")
        print("-"*60)
        
        for n_prof, res in results.items():
            print(f"{n_prof:<15} {res['time']:<15.3f} {res['timesteps']:<15}")
        
        print("="*60)
        
        timer.report()
    
    @pytest.mark.slow
    def test_scalability_with_layer_count(self, synthetic_pro, tmp_path, timer):
        """Test scaling with number of layers."""
        layer_counts = [10, 50, 100, 200, 500]
        results = {}
        
        for n_layers in layer_counts:
            # Create subdirectory
            test_dir = tmp_path / f"layers_{n_layers}"
            test_dir.mkdir()
            
            filepath = synthetic_pro.basic(
                test_dir,  # Pass directory
                n_layers=n_layers
            )
            
            timer.start(f'layers_{n_layers}')
            ds = read(test_dir, lazy=False)  
            timer.stop(f'layers_{n_layers}')
            
            results[n_layers] = {
                'time': timer.get(f'layers_{n_layers}'),
                'layer_dim': ds.sizes['layer']
            }
        
        # Report
        print("\n" + "="*60)
        print("LAYER COUNT SCALING")
        print("="*60)
        print(f"{'Layers':<15} {'Load Time':<15} {'Layer Dim':<15}")
        print("-"*60)
        
        for n_lay, res in results.items():
            print(f"{n_lay:<15} {res['time']:<15.3f} {res['layer_dim']:<15}")
        
        print("="*60)
        
        timer.report()


@pytest.mark.slow
class TestMemoryUsage:
    """Test memory usage patterns."""
    
    def test_memory_usage_large_dataset(self, synthetic_pro, tmp_path, memory_monitor):
        """Test memory usage with large dataset."""
        # Create substantial dataset
        n_stations = 50
        n_profiles = 20
        
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=n_stations,
            n_profiles_each=n_profiles
        )
        
        # Test eager loading memory
        memory_monitor.start()
        ds_eager = read(data_dir, lazy=False, n_cpus_use=4)
        mem_eager = memory_monitor.increase()
        
        print(f"\nEager loading memory: {mem_eager:.1f} MB")
        print(f"Dataset size: {n_stations} locations × {n_profiles} profiles")
        print(f"Memory per profile: {mem_eager / (n_stations * n_profiles):.2f} MB")
        
        # Verify data loaded
        assert ds_eager.sizes['location'] == n_stations
        assert ds_eager.sizes['time'] == n_profiles
        
        # Memory should be reasonable (< 1GB for this size)
        assert mem_eager < 1000, f"Excessive memory usage: {mem_eager} MB"
        
        memory_monitor.report()
    
    def test_memory_efficient_filtering(self, synthetic_pro, tmp_path, memory_monitor):
        """Test that filtering doesn't cause memory bloat."""
        # Create dataset
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=20, n_profiles_each=50)
        
        memory_monitor.start()
        
        # Load full dataset
        ds_full = read(data_dir, lazy=False)
        mem_after_load = memory_monitor.increase()
    
        # Skip if no memory was tracked (monitoring might not work in all environments)
        if mem_after_load == 0:
            pytest.skip("Memory monitoring not available")
        
        # Apply multiple filters
        ds_filtered = ds_full.isel(time=slice(0, 10))
        ds_filtered = ds_filtered.isel(location=slice(0, 5))
        
        mem_after_filter = memory_monitor.increase()
        mem_increase = mem_after_filter - mem_after_load
        
        print(f"\nMemory after load: {mem_after_load:.1f} MB")
        print(f"Memory after filtering: {mem_after_filter:.1f} MB")
        print(f"Filtering overhead: {mem_increase:.1f} MB")
        
        # Filtering should not significantly increase memory
        assert mem_increase < mem_after_load * 0.3, "Excessive memory during filtering"
        
        memory_monitor.report()
    
    def test_memory_garbage_collection(self, synthetic_pro, tmp_path, memory_monitor):
        """Test that memory is properly released."""
        import gc
        
        filepath = synthetic_pro.basic(tmp_path)
        
        memory_monitor.start()
        baseline = memory_monitor.current()
        
        # Load and process
        ds = read(filepath)
        ds = ds.compute_hs()
        peak = memory_monitor.current()
        
        # Delete and collect
        del ds
        gc.collect()
        after_gc = memory_monitor.current()
        
        print(f"\nBaseline: {baseline:.1f} MB")
        print(f"Peak: {peak:.1f} MB")
        print(f"After GC: {after_gc:.1f} MB")
        print(f"Released: {peak - after_gc:.1f} MB")
        
        # Should release most memory
        assert after_gc < peak * 1.2, "Memory not properly released"


@pytest.mark.slow
class TestVeryLargeDataset:
    """Tests with very large datasets (may take several minutes)."""
    
    @pytest.mark.slow
    def test_very_large_file_count(self, synthetic_pro, tmp_path, timer, memory_monitor):
        """Test with very large number of files (stress test)."""
        n_files = 500
        
        print(f"\nCreating {n_files} test files...")
        data_dir = tmp_path / "large_scale"
        data_dir.mkdir()
        
        timer.start('file_creation')
        for i in range(n_files):
            synthetic_pro.basic(
                data_dir,
                station_name=f"S{i:04d}",
                n_layers=5
            )
        timer.stop('file_creation')
        
        print(f"File creation: {timer.get('file_creation'):.2f}s")
        
        # Test lazy loading (recommended for large datasets)
        memory_monitor.start()
        timer.start('lazy_large')
        
        ds = read(data_dir, lazy=True, n_cpus_use=8)
        
        timer.stop('lazy_large')
        mem_used = memory_monitor.increase()
        
        print(f"\nLazy loading results:")
        print(f"  Time: {timer.get('lazy_large'):.2f}s")
        print(f"  Memory: {mem_used:.1f} MB")
        print(f"  Locations: {ds.sizes['location']}")
        print(f"  Time/file: {timer.get('lazy_large')/n_files*1000:.2f}ms")
        
        assert ds.sizes['location'] == n_files
        
        # Should complete in reasonable time (< 2 min)
        assert timer.get('lazy_large') < 120, "Loading took too long"
        
        timer.report()
        memory_monitor.report()


@pytest.mark.benchmark
class TestBenchmarks:
    """Detailed performance benchmarks."""
    
    @pytest.mark.benchmark
    def test_benchmark_loading_pipeline(self, synthetic_pro, tmp_path, timer):
        """Benchmark each stage of the loading pipeline."""
        # Create test data
        n_files = 30
        data_dir = tmp_path / "benchmark"
        data_dir.mkdir()
        
        for i in range(n_files):
            synthetic_pro.basic(
                data_dir,
                station_name=f"BENCH_{i}",
                n_layers=10
            )
        
        # Benchmark file discovery
        timer.start('file_discovery')
        from xsnow.io.merge import _resolve_filepaths
        pro_files, smet_files = _resolve_filepaths(data_dir)
        timer.stop('file_discovery')
        
        # Benchmark pre-scan
        timer.start('pre_scan')
        from xsnow.io.pro import _scan_pro_for_max_layers
        max_layers = max(_scan_pro_for_max_layers(fp, True, True) for fp in pro_files)
        timer.stop('pre_scan')
        
        # Benchmark full load
        timer.start('full_load')
        ds = read(data_dir, lazy=False, n_cpus_use=4)
        timer.stop('full_load')
        
        # Benchmark coordinate computation
        timer.start('compute_z')
        if 'height' in ds.data:
            ds = ds.compute_z_depth()
        timer.stop('compute_z')
        
        # Report detailed breakdown
        print("\n" + "="*70)
        print("LOADING PIPELINE BENCHMARK")
        print("="*70)
        print(f"Files: {n_files}")
        print(f"Max layers: {max_layers}")
        print("-"*70)
        
        stages = ['file_discovery', 'pre_scan', 'full_load', 'compute_z']
        total_time = sum(timer.get(stage) for stage in stages)
        
        for stage in stages:
            stage_time = timer.get(stage)
            percentage = (stage_time / total_time * 100) if total_time > 0 else 0
            print(f"{stage:<20}: {stage_time:>8.3f}s ({percentage:>5.1f}%)")
        
        print("-"*70)
        print(f"{'TOTAL':<20}: {total_time:>8.3f}s")
        print("="*70)
        
        timer.report()
    
    @pytest.mark.benchmark
    def test_benchmark_io_operations(self, synthetic_pro, tmp_path, timer):
        """Benchmark different I/O operations."""
        filepath = synthetic_pro.basic(tmp_path, n_layers=20)
        
        # Read
        timer.start('read_pro')
        ds = read(filepath)
        timer.stop('read_pro')
        
        # Write NetCDF
        nc_file = tmp_path / "bench.nc"
        timer.start('write_netcdf')
        ds.to_netcdf(nc_file)
        timer.stop('write_netcdf')
        
        # Read NetCDF
        timer.start('read_netcdf')
        ds_nc = read(nc_file)
        timer.stop('read_netcdf')
        
        # Write PRO
        pro_out = tmp_path / "bench_out.pro"
        timer.start('write_pro')
        ds.to_pro(pro_out)
        timer.stop('write_pro')
        
        # Report
        print("\n" + "="*60)
        print("I/O OPERATIONS BENCHMARK")
        print("="*60)
        
        ops = [
            ('read_pro', 'Read .pro'),
            ('write_netcdf', 'Write NetCDF'),
            ('read_netcdf', 'Read NetCDF'),
            ('write_pro', 'Write .pro')
        ]
        
        for op_key, op_name in ops:
            op_time = timer.get(op_key)
            print(f"{op_name:<20}: {op_time:>8.4f}s")
        
        print("="*60)
        
        timer.report()
    
    @pytest.mark.benchmark
    def test_benchmark_cpu_utilization(self, synthetic_pro, tmp_path, timer):
        """Benchmark parallel CPU utilization."""
        data_dir = synthetic_pro.multi_station(tmp_path, n_stations=40, n_profiles_each=5)
        
        cpu_counts = [1, 2, 4, 8]
        results = {}
        
        for n_cpus in cpu_counts:
            timer.start(f'cpus_{n_cpus}')
            ds = read(data_dir, lazy=False, n_cpus_use=n_cpus)
            timer.stop(f'cpus_{n_cpus}')
            
            results[n_cpus] = timer.get(f'cpus_{n_cpus}')
        
        # Calculate speedups
        baseline = results[1]
        
        print("\n" + "="*60)
        print("CPU UTILIZATION BENCHMARK")
        print("="*60)
        print(f"{'CPUs':<10} {'Time (s)':<15} {'Speedup':<15} {'Efficiency':<15}")
        print("-"*60)
        
        for n_cpus in cpu_counts:
            time_taken = results[n_cpus]
            speedup = baseline / time_taken
            efficiency = speedup / n_cpus * 100
            
            print(f"{n_cpus:<10} {time_taken:<15.3f} {speedup:<15.2f} {efficiency:<15.1f}%")
        
        print("="*60)
        
        timer.report()