# Performance Testing & Profiling Guide

Complete guide to benchmarking and memory profiling in the xsnow project.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running Benchmarks](#running-benchmarks)
4. [Memory Profiling](#memory-profiling)
5. [Optimization Workflow](#optimization-workflow)
6. [CI/CD Integration](#cicd-integration)
7. [Tools Reference](#tools-reference)
8. [Examples](#examples)

---

## Quick Start

```bash
# Install dependencies
pip install pytest-benchmark memory-profiler psutil

# Run benchmarks
pytest tests/performance/ --benchmark-only

# Run with memory profiling
pytest tests/performance/ -v --memprof

# Compare before/after optimization
pytest tests/performance/ --benchmark-save=before
# ... make your optimization ...
pytest tests/performance/ --benchmark-compare=before
```

---

## Installation

### Required
```bash
pip install pytest-benchmark
```

### Optional (for memory profiling)
```bash
pip install memory-profiler psutil
```

### Development Setup
```bash
# Add to requirements-dev.txt
pytest-benchmark>=4.0.0
memory-profiler>=0.61.0
psutil>=5.9.0
```

---

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
pytest tests/performance/test_benchmarks.py --benchmark-only

# Run specific benchmark
pytest tests/performance/test_benchmarks.py::TestLoadingBenchmarks::test_benchmark_eager_loading_small --benchmark-only

# Get detailed statistics
pytest tests/performance/ --benchmark-only --benchmark-columns=min,max,mean,stddev,median,ops,rounds
```

### Comparing Performance

```bash
# Save baseline
pytest tests/performance/ --benchmark-only --benchmark-save=baseline

# Make your changes...

# Compare against baseline
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline

# See side-by-side comparison
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:5%
```

### Benchmark Output Example

```
-------------------------------- benchmark: 3 tests --------------------------------
Name (time in ms)                           Min       Max      Mean    Median
-------------------------------------------------------------------------------------
test_benchmark_lazy_loading_small        5.234     6.123     5.456     5.389
test_benchmark_eager_loading_small      12.345    14.567    13.123    13.001
test_benchmark_readlines_approach       45.678    52.345    48.234    47.891
-------------------------------------------------------------------------------------
```

---

## Memory Profiling

### Using Built-in Memory Tracker

```python
from tests.performance.profiling_utils import track_memory

# In your test
def test_memory_usage():
    with track_memory("loading data") as tracker:
        ds = read("data/", lazy=False)
        tracker.snapshot("After load")
        
        ds.compute()
        tracker.snapshot("After compute")
    
    print(tracker.report())
    
    # Assert memory doesn't exceed threshold
    assert tracker.get_increase() < 1000  # < 1GB
```

Output:
```
Memory Profile:
------------------------------------------------------------
Before loading data                RSS:   245.23 MB
After load                         RSS:   567.89 MB
  └─ Δ                                   +322.66 MB
After compute                      RSS:  1234.56 MB
  └─ Δ                                   +666.67 MB
------------------------------------------------------------
Total increase:                          +989.33 MB
```

### Detecting Accidental Computation

```python
from tests.performance.profiling_utils import assert_not_computed

def test_lazy_stays_lazy():
    ds = read("data/", lazy=True)
    
    # This should NOT compute the data
    _ = ds.data.sizes
    _ = ds.data.coords
    
    # Verify still lazy
    assert_not_computed(ds, 'height')  # ✅ Pass
    
    # Now compute
    ds_computed = ds.compute()
    
    # This should fail
    assert_not_computed(ds_computed, 'height')  # ❌ Fail (as expected)
```

### Memory-intensive Tests

```python
@pytest.mark.memprof
def test_memory_bound():
    """Test that memory stays under limit."""
    with track_memory() as tracker:
        # Your memory-intensive operation
        ds = read("large_dataset/", lazy=False)
    
    # Assert memory is reasonable
    mem_increase = tracker.get_increase()
    assert mem_increase < 2000, f"Used {mem_increase:.2f} MB, expected < 2000 MB"
```

---

## Optimization Workflow

### Step 1: Identify Bottleneck

Run profiling to find slow operations:

```bash
# Time profiling
pytest tests/performance/ --benchmark-only --benchmark-sort=mean

# Memory profiling
pytest tests/performance/ -v -s --memprof
```

### Step 2: Create Benchmark

```python
# tests/performance/test_my_optimization.py

def current_implementation():
    # Your current code
    pass

def optimized_implementation():
    # Your optimization
    pass

class TestOptimization:
    def test_bench_current(self, benchmark):
        result = benchmark(current_implementation)
        assert result is not None
    
    def test_bench_optimized(self, benchmark):
        result = benchmark(optimized_implementation)
        assert result is not None
```

### Step 3: Baseline

```bash
pytest tests/performance/test_my_optimization.py --benchmark-save=before
```

### Step 4: Optimize

Make your changes...

### Step 5: Compare

```bash
pytest tests/performance/test_my_optimization.py --benchmark-compare=before

# Fail if performance regresses by >10%
pytest tests/performance/test_my_optimization.py --benchmark-compare=before --benchmark-compare-fail=mean:10%
```

### Step 6: Document

```python
"""
Performance optimization: Use generator instead of readlines()

Benchmark results:
- Before: 45.234 ms ± 2.1 ms
- After:  23.456 ms ± 1.2 ms  
- Speedup: 1.93x

Memory results:
- Before: 150 MB
- After:  45 MB
- Savings: 105 MB (70% reduction)
"""
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest-benchmark memory-profiler psutil
      
      - name: Run benchmarks
        run: |
          pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json
      
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
      
      - name: Performance regression check
        run: |
          # Compare against main branch baseline
          git fetch origin main
          git checkout origin/main -- .benchmarks/
          pytest tests/performance/ --benchmark-only --benchmark-compare=0001 --benchmark-compare-fail=mean:20%
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Run fast benchmarks before commit
pytest tests/performance/test_benchmarks.py::TestLoadingBenchmarks::test_benchmark_eager_loading_small --benchmark-only

if [ $? -ne 0 ]; then
    echo "❌ Performance tests failed!"
    exit 1
fi

echo "✅ Performance tests passed"
```

---

## Tools Reference

### profiling_utils.py

#### track_memory()
```python
with track_memory("operation") as tracker:
    # Your code
    tracker.snapshot("checkpoint")

print(tracker.report())
tracker.assert_no_leak(threshold_mb=100)
```

#### Timer
```python
with Timer("my operation") as t:
    # Your code
    pass

print(f"Took {t.elapsed:.4f}s")
```

#### Benchmark
```python
bench = Benchmark()
bench.run("approach_a", func_a, n_runs=100)
bench.run("approach_b", func_b, n_runs=100)
bench.compare("approach_a")
```

#### assert_not_computed()
```python
ds = read("data/", lazy=True)
assert_not_computed(ds, 'height')  # Verify lazy
```

---

## Examples

### Example 1: Compare Parser Approaches

```python
from tests.performance.profiling_utils import Benchmark

def parse_with_readlines(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    return process_lines(lines)

def parse_with_generator(filepath):
    with open(filepath) as f:
        for line in f:
            process_line(line)

# Benchmark
bench = Benchmark()
bench.run("readlines", parse_with_readlines, "data.pro", n_runs=100)
bench.run("generator", parse_with_generator, "data.pro", n_runs=100)
bench.compare("readlines")
```

### Example 2: Memory Leak Detection

```python
def test_no_memory_leak():
    """Ensure repeated operations don't leak memory."""
    
    tracker = MemoryTracker()
    tracker.snapshot("Start")
    
    for i in range(10):
        ds = read("data/")
        process(ds)
        del ds
        gc.collect()
        tracker.snapshot(f"Iteration {i+1}")
    
    # Memory should be stable after first iteration
    first_iter = tracker.snapshots[1]['rss_mb']
    last_iter = tracker.snapshots[-1]['rss_mb']
    increase = last_iter - first_iter
    
    assert increase < 50, f"Memory leak detected: {increase:.2f} MB increase"
```

### Example 3: Lazy vs Eager Decision

```python
def test_lazy_vs_eager_threshold():
    """Find optimal threshold for lazy loading."""
    
    results = {}
    
    for n_files in [5, 10, 20, 50, 100]:
        data_dir = create_test_files(n_files)
        
        # Time eager
        start = time.time()
        ds_eager = read(data_dir, lazy=False)
        time_eager = time.time() - start
        
        # Time lazy
        start = time.time()
        ds_lazy = read(data_dir, lazy=True)
        time_lazy = time.time() - start
        
        results[n_files] = {
            'eager': time_eager,
            'lazy': time_lazy,
            'speedup': time_eager / time_lazy
        }
    
    # Find crossover point
    for n, times in results.items():
        print(f"{n} files: lazy is {times['speedup']:.2f}x {'faster' if times['speedup'] > 1 else 'slower'}")
```

### Example 4: Real-world Optimization

See `test_parser_optimization_example.py` for a complete example of:
- Comparing 3 different implementations
- Memory profiling
- Timing benchmarks
- Automated recommendation

Run it:
```bash
python -m tests.performance.test_parser_optimization_example
```

---

## Best Practices

### 1. Always Baseline First
```bash
pytest tests/performance/ --benchmark-save=before
# Make changes...
pytest tests/performance/ --benchmark-compare=before
```

### 2. Use Appropriate Data Sizes
- Small benchmarks for quick feedback (< 1s)
- Medium benchmarks for realistic scenarios (1-10s)
- Large benchmarks for scalability tests (marked with `@pytest.mark.slow`)

### 3. Isolate Tests
```python
# Bad: Tests interfere with each other
def test_a():
    global_state.data = read("data/")

# Good: Clean state per test
def test_a(tmp_path):
    data = read(tmp_path)
```

### 4. Assert on Performance
```python
def test_fast_enough():
    with Timer() as t:
        result = expensive_operation()
    
    # Don't just measure, assert!
    assert t.elapsed < 5.0, f"Too slow: {t.elapsed:.2f}s"
```

### 5. Document Benchmarks
```python
def test_benchmark_parser():
    """
    Benchmark parser performance.
    
    Target: < 50ms for typical .pro file (100 profiles)
    Baseline: 45.2ms ± 2.1ms (2024-10-30)
    """
    result = benchmark(parse_pro, "typical_file.pro")
```

---

## Troubleshooting

### Benchmarks are Noisy
- Use more runs: `--benchmark-min-rounds=100`
- Disable garbage collection: Already in pytest.ini
- Run on quiet system (close other apps)

### Memory Tests are Flaky
- Call `gc.collect()` before measurements
- Use larger thresholds on CI (±20%)
- Average multiple runs

### Can't Install memory_profiler
- Use lightweight `profiling_utils.py` instead
- Only requires `psutil`: `pip install psutil`

---

## Summary

**Quick Commands:**
```bash
# Development
pytest tests/performance/ --benchmark-only -v

# Before optimization
pytest tests/performance/ --benchmark-save=baseline

# After optimization  
pytest tests/performance/ --benchmark-compare=baseline

# CI/CD
pytest tests/performance/ --benchmark-only --benchmark-json=results.json
```

**Key Tools:**
- `pytest-benchmark`: Timing and statistical analysis
- `profiling_utils.py`: Lightweight memory tracking
- `memory_profiler`: Detailed memory profiling (optional)
- `test_parser_optimization_example.py`: Complete example

**Remember:**
- Benchmark first, optimize second
- Always compare against baseline
- Assert on performance requirements
- Document your optimizations
