# Technical Plan: Adapting `snowpack_reader.py` to the AvaCollabra Data Model

## 1. Introduction:

This document outlines the technical plan for refactoring the existing [`snowpack_reader.py`](https://github.com/ronimos/snowpack/blob/main/src/snowpack_reader.py) script to align with the specifications detailed in the [`development_plan.md`](https://gitlab.com/avacollabra/postprocessing/py-base-initial-dump/-/blob/main/development_plan.md?ref_type=heads). The current reader is a high-performance, single-file parser that produces a time-series dataset for one location. The development plan specifies a more powerful, multi-dimensional data model designed for scalability and interoperability across different data sources (SNOWPACK, CROCUS, etc.).

The core of this work involves shifting from a file-centric paradigm to a data-centric one. This means evolving the reader from a tool that processes one file at a time into a comprehensive package that can manage and analyze gridded snow profile data from multiple locations, slopes, and model realizations simultaneously. This refactoring is essential for achieving the project's goals of scalability, collaborative development, and support for large-scale operational forecasting.

### 1.5. Proposed Project Structure

To facilitate a modular and maintainable codebase, the functionality of the original `snowpack_reader.py` will be broken down into a structured package. This approach separates concerns, making the project easier to develop, test, and extend.

```
xsnow/
├── __init__.py
├── dataset.py          # Defines the core SnowProfileDataset class.
├── io.py               # Contains all file parsers (read_pro, read_smet) and writers (to_pro).
│
├── analysis.py         # Contains high-level analysis functions (find_layer_by_criteria).
│
├── registry.py         # Holds the canonical variable name registry and metadata.
│
└── utils.py            # Optional: For helper functions like vectorized computations.

tests/
├── __init__.py
├── data/               # Contains small sample files for testing.
│   ├── sample.pro
│   └── sample.smet
├── test_io.py          # Tests for all parsing and writing functions.
└── test_analysis.py    # Tests for analysis functions.
```

* **`dataset.py`**: The heart of the package, defining the main user-facing `SnowProfileDataset` class that wraps the `xarray.Dataset`.
* **`io.py`**: All logic related to reading from and writing to files. This keeps data format specifics isolated.
* **`analysis.py`**: High-level scientific and post-processing functions that operate on `SnowProfileDataset` objects.
* **`registry.py`**: Centralizes the mapping of variable names, units, and metadata, ensuring consistency across the package.

This structure directly supports the goals of the development plan by creating a clean, collaborative, and testable environment.

## 2. Core Architectural Changes

The most significant changes are architectural, focusing on adopting the new internal data model.

### 2.1. From `SnowpackProfile` Class to a Data Model Wrapper

The current `SnowpackProfile` class is designed to represent the contents of a single `.pro` file. The development plan calls for a new class implementation that **wraps an `xarray.Dataset`** via composition.

**Required Work:**

1.  **Create a New Wrapper Class:** Introduce a new primary class (e.g., `SnowProfileDataset`) that will contain the `xarray.Dataset` as its main attribute. This class will be the main object users interact with.
2.  **Deprecate `SnowpackProfile` as a User-Facing Object:** The logic within `SnowpackProfile` for parsing files will be extracted and moved into dedicated parser functions (see Section 3). The class itself will be retired or repurposed as an internal utility.

### 2.2. Adopting the Multi-Dimensional Data Model

The development plan's central feature is a gridded data structure with five primary dimensions: `(location, time, slope, realization, layer)`. The current reader only produces a dataset with `(timestamp, layer_index)`.

**Required Work:**

1.  **Modify Parser Outputs:** All parser functions must be updated to produce `xarray.Dataset` objects that conform to the new 5D structure. When reading a single `.pro` file, the `location`, `slope`, and `realization` dimensions will have a size of 1.
2.  **Implement Coordinate Variables:** The parsers must populate the coordinate variables specified in the plan. For example, when parsing a `.pro` file, the `location` dimension will have coordinates for `latitude`, `longitude`, and `altitude` extracted from the file header.
3.  **Handle Data Sparsity:** Implement the specified strategies for handling sparse data, particularly along the `layer` dimension. This involves padding profiles with `NaN` values to a fixed maximum layer count to maintain a regular grid.
4.  **Add Profile Status Tracking:** Introduce the `profile_status` data variable as specified. The parser will need to determine the status (e.g., "valid profile with snow") and record it in the dataset.

## 3. Refactoring Parsing and I/O

The file reading logic needs to be extracted from the `SnowpackProfile` class and structured into the modular `read_*` functions defined in the development plan.

### 3.1. Implementing Standalone `read_*` Functions

The plan specifies a clear API with functions like `read_pro()`, `read_smet()`, and a high-level `read()` wrapper.

**Required Work:**

1.  **Create `read_pro()`:** Extract the line-by-line parsing logic from the `SnowpackProfile._read_profile()` method and place it inside a new `read_pro()` function. This function will take a file path and return a `SnowProfileDataset` object compliant with the new 5D data model. It must also implement the specified arguments like `datetime_start` and `include_smet`.
2.  **Implement Surface Hoar Logic:** As specified in the plan, ensure the `read_pro` function correctly identifies surface hoar information from the file header and represents it as the topmost layer in the profile, filling missing properties with `NaN`.
3.  **Create `read()` Wrapper:** Implement the high-level `read()` function. This function will be responsible for:
    * Iterating through file paths and directories.
    * Auto-detecting the file type and calling the appropriate parser (e.g., `read_pro`).
    * Merging the datasets returned by the individual parsers into a single, consolidated `SnowProfileDataset`. This involves correctly assigning coordinates to the `location` and `realization` dimensions to avoid conflicts.

### 3.2. Standardizing Variable Names

The hardcoded `PARAM_CODES` dictionary in `snowpack_reader.py` must be replaced with the canonical name registry proposed in the development plan.

**Required Work:**

1.  **Create the Name Registry:** Implement a separate module or configuration file for the variable name registry.
2.  **Consider Header Definitions:** The parser must first check for a `[HEADER]` section within the `.pro` file. This section can define custom variable names for the parameter codes. The parser should prioritize these names before falling back to the hardcoded `PARAM_CODES` mapping.
3.  **Update Parsers:** Modify the parsers to use the name registry to translate all found variable names (whether from the header or the hardcoded map) to their canonical names (e.g., "rho", "density" -> "density").
4.  **Attach Metadata:** The parsers must attach the corresponding `standard_name` and `si_units` from the registry as attributes to each data variable in the `xarray.Dataset`.

### 3.3. Implementing `to_*()` Write Methods

To enable interoperability, the `SnowProfileDataset` class will include methods for exporting data into various formats.

**Required Work:**

1.  **Implement `.to_netcdf()` and `.to_zarr()`:** The package will support both NetCDF (for single-file portability) and Zarr (for cloud-native, parallel I/O). These methods will be thin wrappers around `xarray`'s native functionality, ensuring data is correctly converted from GPU to CPU arrays before writing.
2.  **Implement `.to_pro()` and `.to_smet()`:** These methods will convert the multi-dimensional `xarray.Dataset` back into a text-based format. This requires iterating through each profile, formatting the data back to the required string format, and writing each profile to a separate file within a user-specified directory.

### 3.4. Performance Optimizations for I/O

For operational use cases, the I/O functions will include performance-enhancing features.

**Required Work:**

1.  **Incremental Reading of Appended Files:** The `read_pro()` function will support a `datetime_start` argument. When provided, the parser will efficiently scan the file and only perform full, expensive parsing for records newer than the provided timestamp.
2.  **Parallel Reading of Multiple Files:** The high-level `read()` wrapper will include a `parallel` boolean flag. When `True`, it will use a `ProcessPoolExecutor` to distribute the reading of multiple files across all available CPU cores.
3.  **Parser Optimization with Cython:** To maximize ingestion speed, the most performance-critical loops within the `.pro` file parser will be identified and rewritten in Cython. This will compile the Python-like code into highly efficient C code, dramatically reducing the time it takes to parse raw text files.

## 4. Adapting Analysis and Utility Methods

The existing high-level analysis methods will be refactored to work with the new multi-dimensional data model.

### 4.1. Updating Analysis Functions

The methods `get_profile_summary` and `find_layer_by_criteria` need to be adapted.

**Required Work:**

1.  **Generalize Method Logic:** Rewrite these functions to operate on a multi-dimensional `SnowProfileDataset`, using `xarray`'s `groupby()` or `apply_ufunc()` capabilities to perform the analysis for each profile.
2.  **Preserve Vectorization:** The core logic within these functions should remain as vectorized as possible to leverage the performance benefits of `xarray` and NumPy/CuPy.

### 4.2. Vectorized Computations (`rc_flat`, `depth`)

The existing vectorized calculations are a major strength and align well with the new model.

**Required Work:**

1.  **Verify Broadcasting:** These methods will require minimal changes. The primary task is to verify that `xarray` correctly broadcasts the calculations across the new dimensions.

### 4.3. Caching and NetCDF I/O

The existing `.nc` caching mechanism will be preserved and enhanced.

**Required Work:**

1.  **Adapt Caching Logic:** The logic for checking for a cached `.nc` file will be moved into the new `read_netcdf()` function.
2.  **Implement Smart Caching of Derived Variables:** Analysis functions will be updated to check if a computationally expensive result (like `rc_flat`) already exists in the dataset before re-computing it. A method will be provided to save these derived variables back to the cached NetCDF or Zarr store, making them instantly available in future sessions.

### 4.4. Concatenating and Merging Datasets

The package will rely on native `xarray` functionality for combining datasets.

**Required Work:**

1.  **Implement `xr.concat` in `read()` Wrapper:** The `read()` function must use `xr.concat` to combine the list of individual datasets into a single object.
2.  **Document Merging Workflows:** Create tutorials demonstrating how to use `xr.merge`, `.update()`, and `.combine_first()` for common use cases.

### 4.5. Visualization Support

The package will provide helpers to enable easy plotting.

**Required Work:**

1.  **Develop a Profile Plotting Interface:** Create a method (e.g., `.plot.profile()`) to generate a stratigraphy plot for a single profile using a library like NiViz or snowpat.
2.  **Implement 2D Map Reshaping:** Create a helper function to reshape scalar data variables into a 2D grid for map-based plotting.

### 4.6. Extensibility with a Plugin System

To allow users to add their own custom analysis functions, an extension system will be implemented.

**Required Work:**

1.  **Create a Decorator-Based Registry:** Implement a decorator (e.g., `@xsnow.register_analysis`) that allows users to register their own functions.
2.  **Provide an Execution Engine:** Create a method (e.g., `dataset.xsnow.run_analysis()`) that can discover and run these registered functions.

### 4.7. Advanced Performance & Scalability

For very large datasets that may not fit into memory, the package will integrate with Dask.

**Required Work:**

1.  **Enable Out-of-Memory Computation with Dask:** Dask will be added as an optional dependency. The `SnowProfileDataset` class and analysis functions will be designed to work with Dask-backed `xarray` objects. This will enable lazy, out-of-memory, and parallel computation for analysis tasks, allowing the package to scale to massive datasets.

## 5. Summary of Key Tasks

1.  **Architectural Shift:**
    * Create a new `SnowProfileDataset` class to wrap `xarray.Dataset`.
    * Restructure the output of all parsers to a `(location, time, slope, realization, layer)` dimensional model.
2.  **Parser Refactoring:**
    * Extract parsing logic into standalone `read_*()` functions.
    * Implement a high-level `read()` wrapper with parallel processing capabilities.
    * Replace `PARAM_CODES` with a canonical variable name registry.
    * Implement `to_*` methods for writing data to various formats, including Zarr.
3.  **Analysis Method Adaptation:**
    * Rewrite `get_profile_summary` and `find_layer_by_criteria` to use `xarray.groupby()`.
    * Verify that existing vectorized calculations broadcast correctly.
    * Implement a decorator-based plugin system for custom analysis.
    * Integrate Dask for out-of-memory and parallel analysis.
4.  **New Feature Implementation:**
    * Add the `profile_status` data variable.
    * Enrich the dataset with CF-compliant metadata.
    * Implement helpers and documentation for data manipulation and visualization.

## 6. Package Integrity and Testing

A comprehensive testing strategy will be implemented to ensure reliability.

**Required Work:**

1.  **Establish Test Infrastructure:** Create a `tests/` directory with subdirectories for unit tests, integration tests, and sample data.
2.  **Implement Unit Tests:** Write `pytest` unit tests for each core component.
3.  **Implement Integration Tests:** Create tests for workflows that involve multiple components.
4.  **Enable Doctests:** Add example code to docstrings and enable `pytest`'s doctest module.
5.  **Set Up Continuous Integration (CI):** Configure a CI pipeline to automatically run the full test suite.

## 7. Open Source and Community Engagement

Several steps are required to prepare the package for a successful open-source release.

**Required Work:**

1.  **Add a `LICENSE` file:** A permissive license like **MIT** or **Apache 2.0** is recommended.
2.  **Create a `README.md` file:** Explain what the package does, how to install it, and provide a quick-start example.
3.  **Create a `CONTRIBUTING.md` file:** Guide potential contributors on how to set up a development environment and submit changes.
4.  **Adopt a `CODE_OF_CONDUCT.md`:** Establish a welcoming community environment.
5.  **Prepare for Publication:** Create a `pyproject.toml` file and use standard Python packaging tools to publish the package to the Python Package Index (PyPI).