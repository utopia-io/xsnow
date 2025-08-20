# xsnow Package Documentation

This document provides an overview of the core components of the `xsnow` package, a powerful tool for reading, analyzing, and post-processing SNOWPACK model output. It details the data I/O module (`xsnow_io.py`) and the main data object (`xsnowDataset`), along with practical examples of how to use them.

## Installation

The `xsnow` package is designed to be installed easily using `pip`. It supports both standard CPU-based analysis and optional GPU acceleration for enhanced performance.

### Standard Installation (CPU Only)

For most users, the standard installation is sufficient for all analysis tasks. This will install the core package and its essential dependencies like `xarray` and `pandas`.

Open your terminal and run:
```bash
pip install xsnow
```

### Installation with GPU Support

For users with a compatible NVIDIA GPU, `xsnow` can leverage the `CuPy` library for significant performance improvements in computationally intensive tasks. This is handled as an optional dependency.

**Prerequisites:**
- An NVIDIA GPU with a modern CUDA driver installed.
- You do not need to install the full CUDA toolkit; the driver is sufficient.

To install `xsnow` with GPU support, use the following command:
```bash
pip install xsnow[gpu]
```
This command installs the core package plus a pre-compiled, ready-to-use version of `CuPy` that is compatible with modern NVIDIA drivers. The analysis functions within `xsnow` will automatically detect and use the GPU if it is available.

### Developer Installation

If you wish to contribute to the development of `xsnow`, it is recommended to install the package in "editable" mode from a cloned repository. This allows you to make changes to the source code and have them immediately take effect without needing to reinstall.

```bash
# 1. Clone the repository from GitHub
git clone [https://github.com/your-username/xsnow.git](https://github.com/your-username/xsnow.git)
cd xsnow

# 2. Install in editable mode with all optional dependencies
pip install -e .[gpu,dev]
```
This command installs the package in editable mode (`-e`) and includes both the `gpu` extras and the `dev` extras, which contain testing tools like `pytest`.

## Core Components

The `xsnow` package is built around two primary modules that handle data ingestion and representation.

### `xsnow_io.py`: Data Input/Output

This module is the engine for reading SNOWPACK `.pro` and `.smet` files. It is designed to be efficient and flexible, handling everything from a single file to large collections.

* **`read(source, parallel=True)`**: This is the main, high-level function for loading data.

  * **Versatile Input**: It accepts a single file path, a list of file paths, or a path to a directory. It automatically detects and reads both `.pro` (snow profiles) and `.smet` (meteorological timeseries) files.

  * **Parallel Processing**: It reads multiple `.pro` and `.smet` files concurrently using all available CPU cores, which dramatically speeds up the loading process for large datasets. 

  * **Data Integrity Check**: When reading a mix of file types, it performs a crucial data integrity check. It will raise a `ValueError` if it finds a `.smet` file for a location that does not have a corresponding `.pro` file, preventing data mismatches.

  * **Automatic Merging**: It intelligently combines the data from all files into a single, consolidated `xarray.Dataset`, correctly aligning them along the `location` dimension.

### `dataset.py`: The `xsnowDataset` Class

This is the central, user-facing object in the `xsnow` package. It acts as a smart and convenient wrapper around a standard `xarray.Dataset`.

* **Composition over Inheritance**: This class *contains* an `xarray.Dataset` instance (in `self.data`) rather than inheriting from it. This design choice, known as **composition**, is more robust and flexible. It prevents future updates to `xarray`'s complex internal machinery from breaking this class and gives us full control over the user-facing API.

* **Flexible Initialization**: You can create an `xsnowDataset` object from a file path, a list of files, a directory, or even an existing `xarray.Dataset`. This flexibility is the key to the package's extension architecture.

* **Seamless `xarray` Integration**: The class automatically forwards attribute and method calls (like `.sel()`, `.mean()`, `.plot`, or `.sizes`) to the underlying `xarray.Dataset`. This means you can interact with it just like you would with a normal `xarray` object.

* **Automatic Re-wrapping**: When you perform an operation that returns a new `xarray.Dataset` (e.g., subsetting the data), the result is automatically wrapped back into an `xsnowDataset` instance. This ensures you can chain operations together without losing the convenience of the wrapper class.

## Advanced Features

### The 5D Data Model
The `xsnow` package structures all snow profile data into a consistent, five-dimensional model. Understanding these dimensions is key to effective data analysis.

* **`location`**: The spatial coordinate, representing a specific station or grid point.
* **`time`**: The temporal coordinate, representing the timestamp of a specific profile.
* **`slope`**: Represents different slope configurations (e.g., a 30-degree, south-facing slope).
* **`realization`**: Used for ensemble simulations, where each realization is a different model run.
* **`layer`**: Represents the individual layers within a single snow profile, indexed from the ground up.

### Hardware-Aware Computation (CPU/GPU)
The `xsnow` package is designed to leverage available hardware for optimal performance. The `xsnowDataset` class provides methods to move data between the CPU and GPU, and the functions in `analysis.py` are hardware-aware.

* **`.to_gpu()`**: Moves the dataset to the GPU for accelerated processing with `CuPy`.
* **`.to_cpu()`**: Moves the data back to the CPU for tasks like saving files.

```python
# 1. Load data (on CPU)
ds_cpu = xsnow.xsnowDataset('path/to/data')

# 2. Move to GPU for fast computation
ds_gpu = ds_cpu.to_gpu()

# 3. Analysis functions automatically use the GPU
ds_results_gpu = xsnow.analysis.compute_rc_flat(ds_gpu)

# 4. Move results back to CPU for saving
ds_results_cpu = ds_results_gpu.to_cpu()
ds_results_cpu.to_netcdf('results.nc')
```

### Plotting and Visualization
Because `xsnowDataset` is a wrapper around `xarray.Dataset`, you can use `xarray`'s powerful and convenient built-in plotting capabilities, which are based on `matplotlib`.

**Example: Plotting a single snow profile's density**
```python
import matplotlib.pyplot as plt

ds = xsnow.xsnowDataset('path/to/station_A.pro')

# Select a single profile for a specific time
profile = ds.data.isel(location=0, time=0)

# Use xarray's .plot() method
profile['density'].plot(y='height')

plt.title("Density Profile for Station A")
plt.xlabel("Density (kg m-3)")
plt.ylabel("Height (m)")
plt.grid(True)
plt.show()
```

## Usage Examples

Here are some practical examples of how to use the `xsnow` package.

### 1. Loading Data

You can load data in several ways, depending on how your files are organized.

**Load a single `.pro` file:**

```python
import xsnow

# The xsnowDataset class calls the read() function internally
ds = xsnow.xsnowDataset('path/to/your/station.pro')
print(ds)
```

**Load all `.pro` and `.smet` files from a directory:**

```python
import xsnow

# This will find all .pro and .smet files, read them, and merge them
# where the locations match.
ds = xsnow.xsnowDataset('path/to/data_directory/')
print(ds)
```

### 2. Merging SMET Data

You can add meteorological data from a `.smet` file to an existing `xsnowDataset` object.

```python
import xsnow

# 1. Load your primary snow profile data
ds_profiles = xsnow.xsnowDataset('path/to/station_A.pro')

# 2. Merge the corresponding SMET data
# This reads 'station_A.smet' and merges its variables with ds_profiles
ds_merged = ds_profiles.merge_smet('path/to/station_A.smet')

print("--- Original Variables ---")
print(list(ds_profiles.data.data_vars))

print("\n--- Merged Variables ---")
print(list(ds_merged.data.data_vars)) # Will now include TA, VW, etc.
```

### 3. Caching and Updating Data

A common workflow is to load a large, cached dataset from a NetCDF file and then update it with the most recent data from new `.pro` or `.smet` files.

**Load from Cache and Update:**
```python
import xsnow

# 1. Load the existing dataset from a fast-loading NetCDF cache
ds_cache = xsnow.xsnowDataset('path/to/historical_data.nc')

# 2. Update the dataset with new files
# The .update() method finds the last timestamp in the cache and only
# reads data from the new files that is more recent.
ds_updated = ds_cache.update('path/to/new_data_directory/')

# 3. Save the newly updated data back to the cache for next time
ds_updated.to_netcdf('path/to/historical_data.nc')
```

### 4. Saving and Exporting Data

You can easily save your processed data to various formats.

**Save to NetCDF (Recommended for Caching):**

```python
ds = xsnow.xsnowDataset('path/to/data_directory/')
# ... perform some analysis ...
processed_ds = ds.compute_hs()

# Save the processed data for fast reloading later
processed_ds.to_netcdf('processed_cache.nc')
```

**Export to `.pro` and `.smet` files:**

The export functions include safeguards to prevent you from accidentally writing a huge number of files.

```python
# Create a dataset with 2 locations and 3 timestamps (6 profiles total)
ds = xsnow.xsnowDataset(['station_A.pro', 'station_B.pro'])

# Export to .pro files (will create 6 files)
ds.to_pro('pro_output_directory/')

# This will raise an error because 6 > 5
try:
    ds.to_pro('pro_output_directory/', max_profiles=5)
except ValueError as e:
    print(f"Caught expected error: {e}")

# Export timeseries variables (like HS, TA, VW) to .smet files
# This will create one file per location (2 files)
ds_with_smet = ds.merge_smet(['station_A.smet', 'station_B.smet'])
ds_with_smet.to_smet('smet_output_directory/')
```

### 5. Structured Queries with Selectors

For more complex or reusable queries, the `selectors` module provides a structured, object-oriented way to filter your data.

**Select by Location ID:**
```python
from xsnow.selectors import LocationSelector

ds = xsnow.xsnowDataset('path/to/data_directory/')
selector = LocationSelector(sel='station_B')
subset = xsnowDataset(selector.apply(ds.data))
print(subset)
```

**Select by Geographic Bounding Box:**
```python
from xsnow.selectors import LocationSelector

selector = LocationSelector(
    lat_range=(45.0, 46.0),
    lon_range=(10.0, 11.0),
    crs='EPSG:4326'  # WGS 84
)
subset = xsnowDataset(selector.apply(ds.data))
print(subset)
```

**Chain Selectors for a Complex Query:**
```python
from xsnow.selectors import LocationSelector, TimeSelector

# 1. Define selectors
selector_bbox = LocationSelector(lat_range=(44.9, 45.6))
selector_time = TimeSelector(sel=slice('2025-01-02', '2025-01-03'))

# 2. Apply them sequentially
subset = ds.data
subset = selector_bbox.apply(subset)
subset = selector_time.apply(subset)
final_ds = xsnowDataset(subset)
print(final_ds)
```

## Extending `xsnow` with Custom Functions

The `xsnowDataset` class is designed to be easily extended with your own analysis functions. There are two primary patterns for this, each suited to different needs.

### Method 1: The Plugin System (`@xsnow.register_analysis`)

This method is designed to make your custom function feel like a **built-in part of the `xsnowDataset` object**. It's the recommended approach for creating a library of reusable tools.

* **How it Works:** You write a standard Python function that takes an `xsnowDataset` as its first argument. By adding the `@xsnow.register_analysis` decorator above it, you "sign up" your function with the `xsnow` package. The `xsnowDataset` object automatically discovers this function, allowing you to call it as if it were a built-in method.

* **Example:**
    ```python
    # In a file like 'my_analysis_plugins.py'
    import xsnow
    import xarray as xr
    
    @xsnow.register_analysis
    def calculate_top_slab_density(ds: xsnow.xsnowDataset, depth_cm: float = 20.0) -> xr.DataArray:
        """Calculates the mean density of the top slab of the snowpack."""
        # ... (calculation logic here) ...
        total_height = ds.data['height'].max(dim='layer', skipna=True)
        slab_bottom_height = total_height - (depth_cm / 100.0)
        top_slab_mask = ds.data['height'] >= slab_bottom_height
        top_slab_densities = ds.data['density'].where(top_slab_mask)
        return top_slab_densities.mean(dim='layer', skipna=True)
    
    # --- In your main script or notebook ---
    import xsnow
    # This import is crucial because it runs the decorator code
    import my_analysis_plugins
    
    ds = xsnow.xsnowDataset('path/to/data')
    # Call your function like a built-in method!
    mean_density = ds.calculate_top_slab_density(depth_cm=30)
    print(mean_density)
    ```

### Method 2: The Standalone Function

This approach treats the analysis as an **external operation** that you perform *on* the dataset. It's simpler and more direct, making it ideal for analyses that are specific to a single script or project.

* **How it Works:** You write a standard Python function that accepts an `xsnowDataset` object as an argument. The function performs its calculations on the internal `xarray.Dataset` (`ds.data`) and then, at the very end, it creates and returns a **new** `xsnowDataset` instance containing the results.

* **Example:**
    ```python
    def add_slab_density_variable(ds: xsnow.xsnowDataset, slab_thickness: float = 0.3) -> xsnow.xsnowDataset:
        """
        Calculates the average density of the top slab and adds it as a new
        variable to the dataset.
        """
        if ds.data is None:
            raise ValueError("Input dataset is empty.")
    
        total_height = ds.data['height'].max(dim='layer')
        slab_layers = ds.data['density'].where(ds.data['height'] >= total_height - slab_thickness)
        slab_density = slab_layers.mean(dim='layer')
    
        # Add metadata to the new variable
        slab_density.attrs = {
            'long_name': f'Average density of the top {slab_thickness}m slab',
            'units': 'kg m-3'
        }
    
        # Add the new variable to the xarray.Dataset
        new_xarray_ds = ds.data.assign(slab_density=slab_density)
    
        # The key step: re-wrap the modified xarray.Dataset in our class
        return xsnow.xsnowDataset(new_xarray_ds)
    
    # --- Usage ---
    # ds = xsnow.xsnowDataset(...)
    # extended_ds = add_slab_density_variable(ds)
    # print(extended_ds.data['slab_density'])
    ```

### Which Method Should You Use?

Both patterns are powerful. Here’s a quick guide to help you choose:

| Use the... | When you want... |
| :--- | :--- |
| **Plugin System (`@register_analysis`)** | ...to create a **reusable library** of functions that feel like built-in methods. Prioritize **convenience and a clean, chainable API** for the end-user (e.g., `ds.compute_hs().calculate_top_slab_density()`). |
| **Standalone Function** | ...to perform a **specific analysis** for a single script or project. Prioritize **explicitness and a clear, functional data flow** (`in -> process -> out`). |

## Advanced Analysis Workflow Example

This example demonstrates how to chain together analysis functions to derive a more complex metric: the average density of the snow slab resting above a potential weak layer.

First, we define an extension function that uses the `find_layer_by_criteria` function from the `analysis` module.

```python
from xsnow.analysis import find_layer_by_criteria

def calculate_mean_density_above_weak_layer(ds: xsnowDataset, criteria: str) -> xsnowDataset:
    """
    Calculates the mean density of the snowpack slab above the highest
    layer that meets the specified weak layer criteria.
    """
    # 1. Find all layers that match the criteria
    weak_layers = find_layer_by_criteria(ds, criteria)

    # 2. Find the depth of the highest weak layer in each profile
    highest_weak_layer_depth = weak_layers.data['height'].max(dim='layer')

    # 3. Create a mask for all layers that are *above* the highest weak layer
    slab_mask = ds.data['height'] > highest_weak_layer_depth

    # 4. Calculate the mean density of those layers
    mean_density_above_wl = ds.data['density'].where(slab_mask).mean(dim='layer')
    
    mean_density_above_wl.attrs = {
        'long_name': f'Mean density above highest weak layer ({criteria})',
        'units': 'kg m-3'
    }

    # 5. Add the new variable to the dataset and re-wrap
    new_xarray_ds = ds.data.assign(mean_density_above_wl=mean_density_above_wl)
    return xsnowDataset(new_xarray_ds)
```

Now, we can use this function in a complete analysis workflow.

```python
from xsnow.dataset import xsnowDataset
from xsnow.analysis import compute_rc_flat

# 1. Load the initial data
ds = xsnowDataset('path/to/your/data')

# 2. (Optional) Move to GPU for faster computation
ds_gpu = ds.to_gpu()

# 3. Compute ram resistance on the GPU
ds_with_rc = compute_rc_flat(ds_gpu)

# 4. Define weak layer criteria and find those layers
criteria = "density < 200"
weak_layers_ds = find_layer_by_criteria(ds_with_rc, criteria)

# 5. Use our extension to calculate the slab density above those weak layers
ds_final = calculate_mean_density_above_weak_layer(ds_with_rc, criteria)

# 6. Move the final result back to the CPU to print or plot
ds_final_cpu = ds_final.to_cpu()
print(ds_final_cpu.data['mean_density_above_wl'])