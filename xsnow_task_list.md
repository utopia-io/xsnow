# Task List – xsnow Package Development

## 1. Package Architecture & Setup
- [x] Create new xsnowDataset class wrapping `xarray.Dataset` (composition, not inheritance).
- [x] Implement modular package structure:
  - `dataset.py` – main data object.
  - `xsnow_io.py` – all read/write parsers.
  - `analysis.py` – post-processing functions.
  - `registry.py` – canonical variable name registry.
  - `selectors.py` – structured query classes.
- [x] Retire `SnowpackProfile` as user-facing class.
- [x] Implement dunder forwarding (`__getitem__`, `__setitem__`) so `xsnowDataset` "feels-like-xarray".
- [x] Move write routines (`to_pro`, etc.) into `xsnow_io.py` and register them as methods via a decorator.
- [x] Rename decorator from `register_analysis()` to `register_xsnowDataset_method()`.
- [x] Add a `to_xarray()` method to return the underlying `xarray.Dataset`.
- [x] Move `cupy` to optional dependencies in `pyproject.toml` to fix installation issues.

## 2. Internal Data Model
- [x] Adopt 5D model: (`location, time, slope, realization, layer`).
- [x] Add coordinate variables: `latitude, longitude, altitude, inclination, azimuth`.
- [x] Re-implement `height` to `z` as the main vertical grid.
    - Status: In progress. Proving difficult due to the multi-dimensional nature of `z`.
- [x] Change `location` to an integer index with names stored as attributes.
- [x] Implement `slope` as an integer index with coordinates `azimuth` and `inclination`.
- [x] Handle sparse `layer` dimension with NaN padding (default, auto, or user-specified max layers).
- [x] Attach CF-compliant metadata to dataset (including CRS).  
    - Still need to double-check that it’s fully complete.
- [x] Rename `param_codes` to `snp_codes` in the registry for clarity.
- [x] Implement `profile_status` variable with defined integer mapping.

## 3. Parsing & I/O
- [x] Implement `read_pro()`:
  - Parse `.pro` files to new data model.
  - Enforce surface hoar as top layer with NaN filling.
  - Populate coordinates and metadata.
- [x] Implement `read_smet()` parser.
- [x] Implement high-level `read()` wrapper:
  - Prefer `read()` for loading data instead of `xsnowDataset()`.
  - Signature updated to accept `Selector` objects and `**parser_kwargs`.
  - Detect file types (`.pro`, `.smet`, `.nc`).
  - Handle multiple files/directories.
  - Merge datasets with data integrity checks.
  - Support parallel reading for `.pro` and `.smet`.
- [x] Implement `read_netcdf()`.
- [x] Add skeletons for `read_crocus()`, `read_caaml()`, `read_json()`, `read_zarr()`.
- [x] Replace `PARAM_CODES` with centralized `registry.py`.
- [x] Implement `.to_pro()`, `.to_smet()`, `.to_netcdf()` methods.
- [x] Add skeletons for `.to_crocus()`, `.to_caaml()`, `.to_json()`, `to_zarr()`.
- [x] Optimize I/O:
  - Incremental reading (`datetime_start`).
- [?] Future: Consider Cythonizing hot loops for `.pro` parsing.  
    - Currently, the reading code doesn’t appear to have enough intensive loops to justify the effort.

## 4. Analysis & Utilities
- [x] Refactor find_layer_by_criteria for 5D datasets.
- [x] Verify and adapt vectorized computations (`compute_critical_cut_length`).
- [x] Implement plugin system for user-defined analysis functions (`@xsnow.register_xsnowDataset_method`).
- [ ] Future: Integrate optional Dask support for large datasets.
- [ ] Future: Provide `.plot.profile()` and map reshaping helpers.
- [ ] Future: Critically test in-memory limitations of sparse matrices.

## 5. Data Manipulation & Postprocessing
- [x] Implement subsetting and filtering via `selectors.py` and `__getattr__`.
- [x] Document and support `xr.concat`, `xr.merge`, and implement `.update()` workflow.
- [x] Add example snow-cover postprocessing functions (`compute_hs`, `compute_hn24`).
- [x] Provide tabular inspection methods via `xarray` and `.to_dataframe()`.

## 6. Documentation & Tutorials
- [x] Maintain continuously updated xsnow Package Documentation page.
- [x] Add "Installation" section to documentation.
- [?] Write docstrings with runnable examples.
- [ ] Build docs/ with Sphinx/Jupyter-Book + tutorials.
- [x] Include example datasets for documentation.
- [x] Add minimal test data to the git repository for use in examples.

## 7. Testing & CI
- [x] Set up tests/ with:
  - Unit tests for each module (`test_io.py`, `test_dataset.py`, `test_selectors.py`).
  - Integration tests for workflows.
  - Sample data created by fixtures.
- [ ] Future: Enable doctests for examples.
- [ ] Future: Configure CI to run full test suite on every commit.

## 8. Open Source Readiness
- [ ] Add LICENSE (MIT or Apache 2.0).
- [ ] Create README.md with install & quick start.
- [ ] Create CONTRIBUTING.md and `CODE_OF_CONDUCT.md`.
- [ ] Package for `PyPI` (with `pyproject.toml`).