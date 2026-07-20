## xsnow Testing Suite Overview

The xsnow testing suite is organized to ensure robust testing of core functionalities, I/O operations, data manipulation, scientific calculations, and performance. Layout follows a layered structure (`unit/`, `integration/`, `performance/`) mirroring `src/xsnow/` where cohesive, rather than the previous topic/dev-progress directories (issue #25).

## Directory Structure & Description

### 1. Root Level

`conftest.py`: Contains shared pytest fixtures used across all test modules, including synthetic data generation and cloning real data repositories.

`test_quick_check.py`: A lightweight sanity check to verify that the testing environment and basic package imports are working correctly.

`quick_test.py`: A standalone debug script (not collected by pytest — doesn't match the `test_*.py` pattern). Not part of the test suite proper.

### 2. Unit (`tests/unit/`)

Mirrors `src/xsnow/` layout. Tests fundamental data structures, internal utilities, and individual modules in isolation.

`test_core_dataset.py`: Validates the main `xsnowDataset` class, including initialization, method forwarding, and property access.

`test_core_decorator_chain.py`: Tests `DatasetDecorator` chaining behavior.

`test_core_profile_status.py`: Validates profile status calculations.

`test_core_type_stubs.py`: Validates `core.pyi` type stubs.

`test_accessors.py`: Tests custom accessors (e.g., `.tz`, `.icssg`) attached to xarray objects.

`test_combiners.py`: Tests combining and merging multiple datasets.

`test_graincode_converter.py`: Tests conversion of grain codes.

`test_multi_slope.py`: Tests handling of datasets with multiple slope angles (cross-cutting: combiners, selectors, I/O).

`test_selectors.py`: Validates selection logic (Time, Location, Slope, etc.).

`test_soil.py`: Tests logic for removing soil layers from snow profiles (registry-registered dataset method).

`test_surface_hoar.py`: Validates handling of surface hoar layers (registry-registered dataset method).

`test_units.py`: Tests the `units` module.

`test_utils.py`: Verifies utility functions such as logging setup and helper methods.

#### 2a. Unit / I/O (`tests/unit/io/`)

Focuses on input/output operations, file parsing, and data serialization (`src/xsnow/io/`). Kept topical rather than forced 1:1 with source filenames — several tests here are cross-cutting across `io/` submodules and splitting them would require rewriting test bodies, not just moving them (see project memory `test-suite-refactor-25` for the reasoning).

`test_api.py`: Tests the high-level `read()` API and general I/O workflows.

`test_base.py`: Tests coordinate/name sanitization helpers in `io/base.py`.

`test_readers.py`: Specialized tests for reading different file formats (`.pro`, `.smet`, NetCDF).

`test_writers.py`: Validates data export functionality to various formats.

`test_parsers.py`: Tests low-level parsing logic for extracting data from raw files.

`test_caching.py`: Ensures the caching mechanism correctly stores and retrieves data.

`test_incremental.py`: Tests logic for incrementally updating existing datasets with new data.

`test_read_conflicts.py`: Tests strict PRO/SMET read conflict handling.

`test_zarr_operations.py`: Tests Zarr-backed I/O operations.

`test_pro_units.py`, `test_smet_units.py`, `test_netcdf_zarr_units.py`: Unit conversion/validation tests per file format.

`test_modes.py`: Compares eager vs. lazy loading to ensure consistency.

`test_eager.py`: Specific tests for eager (in-memory) data loading.

`test_lazy.py`: Specific tests for lazy (Dask-based) loading logic.

`test_parallel.py`: Tests parallel processing capabilities using multiprocessing.

`test_dask.py`: Synthetic tests for Dask integration.

`test_file_validation.py`: Tests file validation logic during loading.

`test_trim_padding.py`: Tests trimming/padding behavior during loading.

#### 2b. Unit / Extensions (`tests/unit/extensions/`)

Tests `src/xsnow/extensions/` modules.

`test_classification.py`: Tests snow classification algorithms.

`test_critical_crack_length.py`: Tests stability metrics like critical crack length.

### 3. Integration (`tests/integration/`)

End-to-end tests that verify multiple components working together.

`test_real_data.py`: Tests pipelines using actual real-world data files.

`test_workflows.py`: Comprehensive workflow tests covering loading, processing, and saving.

`test_validation_and_trimming.py`: End-to-end validation and trimming behavior.

`test_dask_real.py`: Integration tests using Dask with real-world data (moved from `loading/` — heavy real-data end-to-end coverage across soil removal, surface hoar, slope filtering, etc., rather than a Dask-specific unit test).

### 4. Performance (`tests/performance/`)

Benchmarking and scalability tests. Unchanged by the #25 reorg.

`profiling_utils.py`: Utilities for measuring performance.

`setup_performance.py`: Setup script for performance tests.

`test_benchmarks.py`: General benchmark tests.

`test_scalability.py`: Tests scaling with larger datasets.

`test_optimization.py`: Optimization examples and tests.
