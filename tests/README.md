# xsnow Test Suite

Comprehensive test suite for the xsnow package, covering I/O operations, data loading strategies, feature functionality, and real-world workflows.

## Quick Start

### Installation

```bash
# Install test dependencies
pip install -e .[test]

# Or install dev dependencies
pip install -e .[dev]
```

### Run All Tests

```bash
# Run complete test suite
pytest tests/

# Run with coverage report
pytest --cov=xsnow --cov-report=html tests/
```

### Run Specific Categories

```bash
# Integration tests (real data + workflows)
pytest tests/integration/

# Feature tests (selectors, soil removal, etc.)
pytest tests/features/

# Loading tests (eager/lazy modes)
pytest tests/loading/

# I/O tests (parsers, readers, writers)
pytest tests/io/
```

## Test Organization

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── data/                       # Real SNOWPACK data (auto-cloned)
├── io/                         # I/O and parsing tests
├── loading/                    # Loading mode tests
│   ├── test_eager_loading.py
│   ├── test_lazy_loading.py
│   ├── test_loading_modes.py
│   └── test_parallel.py
├── integration/                # Integration and performance tests
│   ├── test_real_data.py
│   ├── test_workflows.py
│   └── test_performance.py
└── features/                   # Feature-specific tests
    ├── test_selectors.py
    ├── test_soil_removal.py
    ├── test_surface_hoar.py
    └── test_multi_slope.py
```

## Test Markers

Filter tests using pytest markers:

```bash
# Real SNOWPACK data tests
pytest -m real_data

# Synthetic/generated data tests
pytest -m synthetic

# Quick tests only (skip slow ones)
pytest -m "not slow"

# Performance benchmarks
pytest -m benchmark

# Edge cases and error handling
pytest -m edge_case
```

## Test Data

### Real Data (Automatic)

Real SNOWPACK model output is automatically cloned on first test run:

```bash
# Data cloned to: tests/data/
# Source: https://gitlab.com/avalanche-warning/universal/archive/test-data
```

**Contents:**
- `pros/gridded/` - Multi-slope gridded simulations
- `pros/snowobs/` - Observation-driven profiles
- `pits/` - Manual snow pit profiles
- `smets/` - Meteorological forcing data

### Synthetic Data (On-Demand)

Tests generate synthetic data as needed using fixtures:

```python
def test_example(synthetic_pro, tmp_path):
    # Generate basic .pro file
    filepath = synthetic_pro.basic(tmp_path, n_layers=10)
    ds = read(filepath)
    assert ds is not None
```

**Available generators:**
- `synthetic_pro.basic()` - Simple valid profile
- `synthetic_pro.with_soil_layers()` - Profile with soil
- `synthetic_pro.with_surface_hoar()` - Profile with SH
- `synthetic_pro.multi_station()` - Multiple stations
- `synthetic_pro.multi_slope()` - Multi-slope data
- `synthetic_smet.basic()` - Meteorological data

## Common Test Patterns

### Basic Test Structure

```python
import pytest
from xsnow import read

@pytest.mark.synthetic
class TestMyFeature:
    def test_basic_functionality(self, synthetic_pro, tmp_path):
        # Arrange
        filepath = synthetic_pro.basic(tmp_path)
        
        # Act
        ds = read(filepath)
        
        # Assert
        assert ds is not None
        assert 'height' in ds.data
```

### Using Real Data

```python
@pytest.mark.real_data
def test_with_real_data(sample_pro_file, assert_physical_validity):
    ds = read(sample_pro_file)
    
    # Automatic physical validation
    assert_physical_validity(ds)
```

### Performance Testing

```python
@pytest.mark.benchmark
def test_performance(synthetic_pro, tmp_path, timer, memory_monitor):
    data_dir = synthetic_pro.multi_station(tmp_path, n_stations=50)
    
    memory_monitor.start()
    timer.start('load')
    
    ds = read(data_dir, lazy=False)
    
    timer.stop('load')
    
    assert timer.get('load') < 30.0  # < 30 seconds
    assert memory_monitor.increase() < 500  # < 500 MB
```

### Dataset Comparison

```python
def test_consistency(real_file_subset, compare_datasets):
    ds1 = read(real_file_subset, lazy=False)
    ds2 = read(real_file_subset, lazy=True)
    
    # Compare with tolerance
    compare_datasets(ds1, ds2, rtol=1e-6)
```

## Key Fixtures

### Data Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `synthetic_pro` | Generator | Create synthetic .pro files |
| `synthetic_smet` | Generator | Create synthetic .smet files |
| `sample_pro_file` | Path | Single real .pro file |
| `gridded_pros_dir` | Path | Directory with gridded data |
| `real_file_subset` | Path | Small subset (5 files) for testing |

### Utility Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `assert_physical_validity` | Function | Validate physical constraints |
| `compare_datasets` | Function | Compare two datasets |
| `timer` | Object | Time operations |
| `memory_monitor` | Object | Monitor memory usage |

### xarray Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `minimal_xr_dataset` | xr.Dataset | Minimal test dataset |
| `snow_profile_xr_dataset` | xr.Dataset | Realistic snow profile |
| `xsnow_dataset` | xsnowDataset | Wrapped dataset |

## Test Coverage

### Current Status

| Component | Coverage | Status |
|-----------|----------|--------|
| I/O Operations | >90% | ✅ |
| Eager Loading | >85% | ✅ |
| Lazy Loading | >85% | ✅ |
| Selectors | >95% | ✅ |
| Soil Removal | >90% | ✅ |
| Surface Hoar | >90% | ✅ |
| Multi-Slope | >85% | ✅ |
| Workflows | >75% | ✅ |

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=xsnow --cov-report=term-missing tests/

# HTML report
pytest --cov=xsnow --cov-report=html tests/
open htmlcov/index.html

# XML report (for CI)
pytest --cov=xsnow --cov-report=xml tests/
```

## Running Specific Tests

### By File

```bash
# Single file
pytest tests/features/test_selectors.py

# With verbose output
pytest -v tests/features/test_selectors.py

# With very verbose output
pytest -vv tests/features/test_selectors.py
```

### By Class

```bash
pytest tests/features/test_selectors.py::TestTimeSelector
```

### By Test Name

```bash
pytest tests/features/test_selectors.py::TestTimeSelector::test_time_selector_slice
```

### By Pattern

```bash
# All tests with "eager" in name
pytest -k eager

# All tests with "selector" in name
pytest -k selector

# Combine patterns
pytest -k "eager and not slow"
```

## Debugging Failed Tests

### Verbose Output

```bash
# Show test names
pytest -v

# Show test names + details
pytest -vv

# Show print statements
pytest -s
```

### Interactive Debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace
```

### Detailed Failure Info

```bash
# Show local variables on failure
pytest -l

# Show full diff on assertion failures
pytest -vv

# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3
```

## Performance Testing

### Quick Benchmarks

```bash
# Run only benchmark tests
pytest -m benchmark tests/integration/test_performance.py

# With timing output
pytest --durations=10 tests/
```

### Scalability Tests

```bash
# File count scaling
pytest tests/integration/test_performance.py::TestScalability::test_scalability_with_file_count

# CPU utilization
pytest tests/integration/test_performance.py::TestBenchmarks::test_benchmark_cpu_utilization
```

### Memory Profiling

```bash
# Memory usage tests
pytest tests/integration/test_performance.py::TestMemoryUsage
```

## CI/CD Integration

### GitHub Actions

```yaml
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
    
    - name: Install dependencies
      run: pip install -e .[test]
    
    - name: Run tests
      run: pytest tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hook

```bash
# Install pre-commit
pip install pre-commit

# Set up hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
pytest -m "not slow" tests/
EOF

chmod +x .git/hooks/pre-commit
```

## Writing New Tests

### 1. Choose Location

```
Unit test → tests/features/
Integration → tests/integration/
Performance → tests/integration/test_performance.py
I/O test → tests/io/
```

### 2. Use Template

```python
import pytest
from xsnow import read

@pytest.mark.synthetic  # Add appropriate marker
class TestNewFeature:
    """Test description."""
    
    def test_basic_case(self, synthetic_pro, tmp_path):
        """Test basic functionality."""
        # Arrange
        filepath = synthetic_pro.basic(tmp_path)
        
        # Act
        ds = read(filepath)
        
        # Assert
        assert ds is not None
    
    def test_edge_case(self, synthetic_pro, tmp_path):
        """Test edge case."""
        # ...
```

### 3. Add Markers

```python
@pytest.mark.synthetic  # Uses synthetic data
@pytest.mark.real_data  # Uses real data
@pytest.mark.slow       # Takes >5 seconds
@pytest.mark.benchmark  # Performance test
@pytest.mark.edge_case  # Edge case/error handling
```

### 4. Document

```python
def test_my_feature(self):
    """
    Test specific aspect of feature.
    
    This test validates that [expected behavior] works correctly
    when [conditions]. It covers [specific scenarios].
    """
```

## Troubleshooting

### Test Data Not Found

```bash
# Manually trigger test data clone
python tests/conftest.py
```

### Import Errors

```bash
# Reinstall in editable mode
pip install -e .

# Or with test dependencies
pip install -e .[test]
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow" tests/

# Run only quick tests
pytest --durations=0 tests/ | grep "0.0[0-9]s"
```

### Memory Issues

```bash
# Run tests sequentially (no parallel)
pytest -n 0 tests/

# Limit memory-intensive tests
pytest -m "not slow" tests/
```

## Best Practices

### ✅ Do

- Use appropriate markers (`@pytest.mark.synthetic`, etc.)
- Use fixtures for setup/teardown
- Write descriptive test names
- Test edge cases and error conditions
- Use `assert_physical_validity` for data validation
- Document complex test logic
- Keep tests independent and isolated

### ❌ Don't

- Hard-code file paths
- Leave print statements in committed code
- Write tests that depend on execution order
- Test multiple features in one test
- Ignore warnings or failures
- Commit large test data files

## Resources

- **Full Documentation:** See `TEST_SUITE_IMPLEMENTATION_SUMMARY.md`
- **Fixture Reference:** Run `pytest --fixtures tests/`
- **Marker Reference:** Run `pytest --markers`
- **pytest Documentation:** https://docs.pytest.org/

## Getting Help

1. Check this README
2. Review `TEST_SUITE_IMPLEMENTATION_SUMMARY.md`
3. Examine `conftest.py` for fixture details
4. Look at existing tests as examples
5. Run `pytest --fixtures` for fixture documentation

## Contributing

When adding tests:

1. Follow existing patterns and conventions
2. Add appropriate markers
3. Use fixtures for data and utilities
4. Write clear docstrings
5. Ensure tests pass locally before committing
6. Update this README if adding new patterns

---

**Quick Reference Card**

```bash
# Common commands
pytest tests/                          # Run all tests
pytest -m "not slow"                   # Quick tests only
pytest -k selector                     # Tests matching "selector"
pytest --cov=xsnow tests/             # With coverage
pytest -x tests/                       # Stop on first failure
pytest -v tests/features/             # Verbose output
pytest --pdb tests/                    # Debug on failure

# Specific test categories
pytest tests/integration/              # Integration tests
pytest tests/features/                 # Feature tests
pytest -m real_data                    # Real data tests
pytest -m benchmark                    # Benchmarks
```

---

**Last Updated:** 2025-01-13  
**Test Framework:** pytest 7.x+  
**Python Version:** 3.9+