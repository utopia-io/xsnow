## xsnow Testing Suite Overview

The xsnow testing suite is organized to ensure robust testing of core functionalities, I/O operations, data manipulation, scientific calculations, and performance. The tests are categorized into logical directories to separate concerns and improve maintainability.

## Directory Structure & Description

### 1. Root Level

The root level contains configuration and quick-check scripts.

`conftest.py`:  Contains shared pytest fixtures used across multiple test modules. This includes setup for synthetic data generation and cloning real data repositories.

`test_quick_check.py`: A lightweight sanity check to verify that the testing environment and basic package imports are working correctly.

### 2. Core (`tests/core/`)

Tests the fundamental data structures and internal utilities of the library.

`test_dataset.py` (formerly `test_core.py`): Validates the main `xsnowDataset` class, including initialization, method forwarding, and property access.

`test_accessors.py`: Tests custom accessors (e.g., `.tz`, `.icssg`) attached to xarray objects.

`test_utils.py`: Verifies utility functions such as logging setup and helper methods.

### 3. I/O (`tests/io/`)

Focuses on input/output operations, file parsing, and data serialization.

`test_api.py` (formerly `test_io.py`): Tests the high-level `read()` API and general I/O workflows.

`test_readers.py`: specialized tests for reading different file formats (`.pro`, `.smet`, `NetCDF`).

`test_writers.py`: Validates data export functionality to various formats.

`test_parsers.py`: Tests low-level parsing logic for extracting data from raw files.

`test_caching.py`: Ensures the caching mechanism correctly stores and retrieves data.

`test_incremental.py` (formerly `test_incremental_update.py`): Tests logic for incrementally updating existing datasets with new data.

### 4. Loading (`tests/loading/`)

Validates different data loading strategies, including lazy loading and parallel processing.

`test_modes.py` (formerly `test_loading_modes.py`): Compares eager vs. lazy loading to ensure consistency.

`test_eager.py` (formerly `test_eager_loading.py`): Specific tests for eager (in-memory) data loading.

`test_lazy.py` (formerly `test_lazy_loading.py`): Specific tests for lazy (Dask-based) loading logic.

`test_parallel.py`: Tests parallel processing capabilities using multiprocessing.

`test_dask.py` (formerly `test_dask_integration.py`): Synthetic tests for Dask integration.

`test_dask_real.py` (formerly `test_dask_integration_real.py`): Integration tests using Dask with real-world data.

### 5. Operations (`tests/ops/`)

Tests data manipulation, selection, and merging capabilities.

`test_selectors.py`: Validates selection logic (Time, Location, Slope, etc.).

`test_combiners.py`: Tests combining and merging multiple datasets.

`test_coordinates.py`: Ensures correct handling and sanitization of coordinates.

`test_multi_slope.py`: Specific tests for handling datasets with multiple slope angles.

### 6. Science (`tests/science/`)

Focuses on scientific algorithms and domain-specific logic.

`test_soil.py` (formerly `test_soil_removal.py`): Tests logic for removing soil layers from snow profiles.

`test_surface_hoar.py`: Validates handling of surface hoar layers.

`test_grains.py` (formerly `test_graincode_converter.py`): Tests conversion of grain codes.

`test_profile_status.py`: Validates profile status calculations.

`test_classification.py`: Tests snow classification algorithms.

`test_stability.py` (formerly `test_critical_crack_length.py`): Tests stability metrics like critical crack length.

### 7. Integration (`tests/integration/`)

End-to-end tests that verify multiple components working together.

`test_real_data.py`: Tests pipelines using actual real-world data files.

`test_workflows.py`: comprehensive workflow tests covering loading, processing, and saving.

### 8. Performance (`tests/performance/`)

Benchmarking and scalability tests.

`profiling_utils.py`: Utilities for measuring performance.

`setup_performance.py` (formerly `setup_performance_testing.py`): Setup script for performance tests.

`test_benchmarks.py`: General benchmark tests.

`test_scalability.py` (formerly `test_performance.py`): Tests scaling with larger datasets.

`test_optimization.py` (formerly `test_parser_optimization_example.py`): Optimization examples and tests.