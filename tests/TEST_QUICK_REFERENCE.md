# xsnow Test Suite - Quick Reference Card

## Installation & Setup

```bash
# Install with test dependencies
pip install -e .[test]

# Verify installation
pytest --version

# First run (clones test data)
pytest tests/ --collect-only
```

## Basic Commands

| Command | Description |
|---------|-------------|
| `pytest tests/` | Run all tests |
| `pytest tests/ -v` | Verbose output |
| `pytest tests/ -vv` | Very verbose |
| `pytest tests/ -s` | Show print statements |
| `pytest tests/ -x` | Stop on first failure |
| `pytest tests/ --lf` | Run last failed |
| `pytest tests/ --ff` | Failed first, then others |

## Test Categories

| Command | Category |
|---------|----------|
| `pytest tests/io/` | I/O operations |
| `pytest tests/loading/` | Loading modes |
| `pytest tests/features/` | Feature tests |
| `pytest tests/integration/` | Integration tests |

## Test Markers

| Marker | Command | Description |
|--------|---------|-------------|
| `synthetic` | `pytest -m synthetic` | Synthetic data tests |
| `real_data` | `pytest -m real_data` | Real SNOWPACK data |
| `slow` | `pytest -m slow` | Slow tests (>5s) |
| `benchmark` | `pytest -m benchmark` | Performance tests |
| `edge_case` | `pytest -m edge_case` | Edge cases |

```bash
# Quick tests only
pytest -m "not slow" tests/

# Real data only
pytest -m real_data tests/

# Synthetic only
pytest -m synthetic tests/
```

## Coverage

```bash
# Terminal report
pytest --cov=xsnow tests/

# HTML report
pytest --cov=xsnow --cov-report=html tests/
open htmlcov/index.html

# With missing lines
pytest --cov=xsnow --cov-report=term-missing tests/

# XML for CI
pytest --cov=xsnow --cov-report=xml tests/
```

## Filtering Tests

```bash
# By file
pytest tests/features/test_selectors.py

# By class
pytest tests/features/test_selectors.py::TestTimeSelector

# By test name
pytest tests/features/test_selectors.py::TestTimeSelector::test_time_selector_slice

# By keyword pattern
pytest -k "selector" tests/
pytest -k "eager and not slow" tests/
```

## Debugging

```bash
# Show local variables on failure
pytest -l tests/

# Drop into debugger on failure
pytest --pdb tests/

# Start debugger at test start
pytest --trace tests/

# Show full diff
pytest -vv tests/

# Show slowest tests
pytest --durations=10 tests/
```

## Fixtures Reference

### Data Fixtures

```python
def test_with_synthetic(synthetic_pro, tmp_path):
    filepath = synthetic_pro.basic(tmp_path, n_layers=10)
    ds = read(filepath)

def test_with_real(sample_pro_file):
    ds = read(sample_pro_file)

def test_with_real_subset(real_file_subset):
    ds = read(real_file_subset, lazy=False)
```

### Utility Fixtures

```python
def test_physical_validity(sample_pro_file, assert_physical_validity):
    ds = read(sample_pro_file)
    assert_physical_validity(ds)

def test_comparison(compare_datasets):
    ds1 = read(source1)
    ds2 = read(source2)
    compare_datasets(ds1, ds2, rtol=1e-6)

def test_performance(timer, memory_monitor):
    timer.start('operation')
    memory_monitor.start()
    
    # ... do work ...
    
    timer.stop('operation')
    assert timer.get('operation') < 10.0
    assert memory_monitor.increase() < 500
```

## Synthetic Data Generators

```python
# Basic profile
filepath = synthetic_pro.basic(tmp_path, n_layers=5)

# With soil layers
filepath = synthetic_pro.with_soil_layers(tmp_path, n_snow=5, n_soil=2)

# With surface hoar
filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)

# Multiple stations
data_dir = synthetic_pro.multi_station(tmp_path, n_stations=5)

# Multiple slopes
data_dir = synthetic_pro.multi_slope(tmp_path, "STATION", slopes=[0, 30])

# Corrupted file
filepath = synthetic_pro.corrupted(tmp_path, corruption_type='malformed_data')

# SMET file
filepath = synthetic_smet.basic(tmp_path, n_timesteps=24)
```

## Common Test Patterns

### Basic Test

```python
@pytest.mark.synthetic
def test_basic(synthetic_pro, tmp_path):
    filepath = synthetic_pro.basic(tmp_path)
    ds = read(filepath)
    assert ds is not None
```

### Real Data Test

```python
@pytest.mark.real_data
def test_real(sample_pro_file):
    ds = read(sample_pro_file)
    assert ds.sizes['time'] > 0
```

### Performance Test

```python
@pytest.mark.benchmark
def test_perf(synthetic_pro, tmp_path, timer):
    data_dir = synthetic_pro.multi_station(tmp_path, n_stations=50)
    
    timer.start('load')
    ds = read(data_dir, lazy=False)
    timer.stop('load')
    
    assert timer.get('load') < 30.0
```

### Error Handling

```python
def test_error(synthetic_pro, tmp_path):
    filepath = synthetic_pro.corrupted(tmp_path, 'malformed_data')
    
    # Should not crash
    ds = read(filepath)
    
    # Either None or empty
    if ds is not None:
        assert ds.sizes['time'] == 0
```

## Performance Benchmarks

```bash
# File count scaling
pytest tests/integration/test_performance.py::TestScalability::test_scalability_with_file_count

# CPU utilization
pytest tests/integration/test_performance.py::TestBenchmarks::test_benchmark_cpu_utilization

# Memory usage
pytest tests/integration/test_performance.py::TestMemoryUsage::test_memory_usage_large_dataset

# Complete benchmark suite
pytest -m benchmark tests/integration/test_performance.py
```

## CI/CD Quick Setup

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - run: pip install -e .[test]
    - run: pytest tests/
    - uses: codecov/codecov-action@v2
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Tests not found | `pip install -e .` |
| Data not found | `python tests/conftest.py` |
| Slow tests | `pytest -m "not slow"` |
| Import errors | `pip install -e .[test]` |
| Memory issues | `pytest -n 0` (sequential) |

## File Locations

| File | Description |
|------|-------------|
| `tests/README.md` | Complete test guide |
| `TEST_SUITE_IMPLEMENTATION_SUMMARY.md` | Architecture details |
| `tests/conftest.py` | Fixtures and configuration |
| `tests/data/` | Real test data (auto-cloned) |

## Help Commands

```bash
# List all fixtures
pytest --fixtures tests/

# List all markers
pytest --markers

# Show test collection
pytest --collect-only tests/

# Show configuration
pytest --showlocals
```

## Common Workflows

### Before Committing
```bash
pytest -m "not slow" tests/
pytest --cov=xsnow tests/
```

### Full Test Run
```bash
pytest tests/ -v --cov=xsnow --cov-report=html
```

### Debug Failing Test
```bash
pytest tests/path/to/test.py::test_name -vv --pdb
```

### Performance Check
```bash
pytest -m benchmark tests/ --durations=0
```

---

**Need More Help?**

1. Check `tests/README.md`
2. Review `TEST_SUITE_IMPLEMENTATION_SUMMARY.md`
3. Run `pytest --fixtures tests/`
4. Look at existing tests as examples

**Quick Stats:**
- **Tests:** 150+
- **Coverage:** >85%
- **Execution:** <2 min (quick), ~5 min (full)
- **Python:** 3.9+
- **Framework:** pytest 7.x+