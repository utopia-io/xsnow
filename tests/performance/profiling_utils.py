# tests/performance/profiling_utils.py
"""
Lightweight profiling utilities for performance testing.

These provide simple memory and timing profiling without heavy dependencies.
"""

import time
import functools
from contextlib import contextmanager
from typing import Callable, Any, Dict, Optional
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# MEMORY PROFILING
# ============================================================================

class MemoryTracker:
    """Lightweight memory usage tracker."""
    
    def __init__(self):
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil required for memory tracking: pip install psutil")
        self.process = psutil.Process()
        self.snapshots = []
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        mem_info = self.process.memory_info()
        self.snapshots.append({
            'label': label,
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'timestamp': time.time()
        })
        return self.snapshots[-1]
    
    def get_increase(self, from_idx: int = 0, to_idx: int = -1) -> float:
        """Get memory increase between two snapshots (MB)."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[to_idx]['rss_mb'] - self.snapshots[from_idx]['rss_mb']
    
    def report(self) -> str:
        """Generate a memory usage report."""
        if not self.snapshots:
            return "No memory snapshots taken"
        
        lines = ["Memory Profile:"]
        lines.append("-" * 60)
        
        for i, snap in enumerate(self.snapshots):
            label = snap['label'] or f"Snapshot {i}"
            lines.append(f"{label:30s} RSS: {snap['rss_mb']:8.2f} MB")
            
            if i > 0:
                increase = snap['rss_mb'] - self.snapshots[i-1]['rss_mb']
                lines.append(f"{'  └─ Δ':30s}      {increase:+8.2f} MB")
        
        lines.append("-" * 60)
        total_increase = self.snapshots[-1]['rss_mb'] - self.snapshots[0]['rss_mb']
        lines.append(f"{'Total increase:':30s}      {total_increase:+8.2f} MB")
        
        return "\n".join(lines)
    
    def assert_no_leak(self, threshold_mb: float = 10.0):
        """Assert that memory hasn't increased beyond threshold."""
        increase = self.get_increase()
        assert increase < threshold_mb, \
            f"Memory leak detected: increased by {increase:.2f} MB (threshold: {threshold_mb} MB)"


@contextmanager
def track_memory(label: str = "operation"):
    """
    Context manager to track memory usage of a code block.
    
    Usage:
        with track_memory("loading data") as tracker:
            ds = read("data/")
        
        print(tracker.report())
    """
    if not PSUTIL_AVAILABLE:
        raise ImportError("psutil required: pip install psutil")
    
    tracker = MemoryTracker()
    gc.collect()
    tracker.snapshot(f"Before {label}")
    
    try:
        yield tracker
    finally:
        tracker.snapshot(f"After {label}")


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Usage:
        @profile_memory
        def load_data(path):
            return read(path)
        
        result = load_data("data/")
        # Prints memory report
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with track_memory(func.__name__) as tracker:
            result = func(*args, **kwargs)
        
        print(f"\n{tracker.report()}")
        return result
    
    return wrapper


# ============================================================================
# TIMING PROFILING
# ============================================================================

class Timer:
    """Lightweight timer for performance measurement."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f}s")
    
    def report(self) -> str:
        """Get timing report."""
        if self.elapsed is None:
            return "Timer not finished"
        return f"{self.name or 'Operation'}: {self.elapsed:.4f}s"


def profile_time(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @profile_time
        def slow_operation():
            time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__) as timer:
            result = func(*args, **kwargs)
        return result
    
    return wrapper


# ============================================================================
# COMBINED PROFILING
# ============================================================================

@contextmanager
def profile_all(label: str = "operation"):
    """
    Profile both time and memory.
    
    Usage:
        with profile_all("loading data") as prof:
            ds = read("data/")
        
        print(prof['memory'].report())
        print(prof['time'].report())
    """
    prof = {
        'memory': MemoryTracker() if PSUTIL_AVAILABLE else None,
        'time': Timer(label)
    }
    
    if prof['memory']:
        gc.collect()
        prof['memory'].snapshot(f"Start {label}")
    
    start = time.perf_counter()
    
    try:
        yield prof
    finally:
        elapsed = time.perf_counter() - start
        prof['time'].elapsed = elapsed
        
        if prof['memory']:
            prof['memory'].snapshot(f"End {label}")
            print(f"\n{prof['memory'].report()}")
        
        print(f"\nTime: {elapsed:.4f}s")


# ============================================================================
# DASK COMPUTATION DETECTOR
# ============================================================================

def is_computed(data) -> bool:
    """
    Check if data is computed (numpy) or lazy (dask).
    
    Returns:
        True if data is computed (numpy array)
        False if data is lazy (dask array)
    """
    import numpy as np
    
    try:
        import dask.array as da
        if isinstance(data, da.Array):
            return False
    except ImportError:
        pass
    
    return isinstance(data, np.ndarray)


def assert_not_computed(ds, var_name: str = 'height'):
    """
    Assert that a variable in the dataset is still lazy (not computed).
    
    Usage:
        ds = read("data/", lazy=True)
        assert_not_computed(ds)  # Should pass
        
        ds_computed = ds.compute()
        assert_not_computed(ds_computed)  # Should fail
    """
    if var_name not in ds.data:
        raise ValueError(f"Variable '{var_name}' not found in dataset")
    
    data = ds.data[var_name].data
    
    try:
        import dask.array as da
        assert isinstance(data, da.Array), \
            f"Data was computed! Expected dask array, got {type(data).__name__}"
    except ImportError:
        raise ImportError("dask required for lazy computation check")


def count_dask_tasks(data) -> int:
    """
    Count the number of tasks in a dask computation graph.
    
    Higher task count = more complex computation graph
    """
    try:
        import dask.array as da
        if isinstance(data, da.Array):
            return len(data.__dask_graph__())
    except (ImportError, AttributeError):
        pass
    
    return 0


# ============================================================================
# COMPARATIVE BENCHMARKING
# ============================================================================

class Benchmark:
    """Simple benchmark comparison tool."""
    
    def __init__(self):
        self.results = {}
    
    def run(self, name: str, func: Callable, *args, n_runs: int = 10, **kwargs):
        """
        Run a benchmark.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            n_runs: Number of runs
            *args, **kwargs: Arguments to pass to func
        """
        times = []
        
        for _ in range(n_runs):
            gc.collect()
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        self.results[name] = {
            'times': times,
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'runs': n_runs
        }
    
    def compare(self, baseline: str):
        """
        Compare all benchmarks to a baseline.
        
        Args:
            baseline: Name of the baseline benchmark
        """
        if baseline not in self.results:
            raise ValueError(f"Baseline '{baseline}' not found")
        
        baseline_time = self.results[baseline]['mean']
        
        print("\nBenchmark Comparison:")
        print("-" * 70)
        print(f"{'Name':<30} {'Mean':<12} {'vs Baseline':<15} {'Speedup'}")
        print("-" * 70)
        
        for name, result in sorted(self.results.items(), 
                                   key=lambda x: x[1]['mean']):
            mean = result['mean']
            speedup = baseline_time / mean
            vs_baseline = f"{(mean/baseline_time - 1)*100:+.1f}%"
            
            marker = "⭐" if name == baseline else "  "
            print(f"{marker} {name:<28} {mean:>10.4f}s  {vs_baseline:>13}  {speedup:>6.2f}x")
        
        print("-" * 70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Track memory
    print("Example 1: Memory tracking")
    with track_memory("example operation") as tracker:
        # Simulate some work
        data = [0] * 10_000_000  # ~40 MB
        tracker.snapshot("After allocation")
        del data
        gc.collect()
        tracker.snapshot("After cleanup")
    
    print(tracker.report())
    
    # Example 2: Compare approaches
    print("\n\nExample 2: Comparing approaches")
    bench = Benchmark()
    
    def approach_a():
        return sum(range(1000000))
    
    def approach_b():
        return sum([i for i in range(1000000)])
    
    def approach_c():
        return sum(i for i in range(1000000))
    
    bench.run("range", approach_a, n_runs=10)
    bench.run("list_comp", approach_b, n_runs=10)
    bench.run("generator", approach_c, n_runs=10)
    
    bench.compare("range")
