# xsnow Package Documentation

`xsnow` is a Python package for reading, analyzing, and post-processing output from the SNOWPACK model. Designed for researchers, avalanche forecasters, and students, it provides a powerful and intuitive toolkit to transform raw model output into a clean, multi-dimensional, and analysis-ready format. By leveraging the robust `xarray` library, `xsnow` simplifies complex data handling and enables both CPU and GPU-accelerated computations, making it easier to work with snow profile and time-series data.

---

This document provides an overview of the core components of the `xsnow` package, a powerful tool for reading, analyzing, and post-processing SNOWPACK model output. It details the data I/O module (`xsnow_io.py`) and the main data object (`xsnowDataset`), along with practical examples of how to use them.

## Installation

The `xsnow` package is designed to be installed easily using `pip`. It supports both standard CPU-based analysis and optional GPU acceleration for enhanced performance.

### Standard Installation (CPU Only)

For most users, the standard installation is sufficient for all analysis tasks. This will install the core package and its essential dependencies like `xarray` and `pandas`.

Open your terminal and run:
```bash
pip install xsnow
````

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
# 1. Clone the repository from GitLab
git clone https://gitlab.com/avacollabra/postprocessing/xsnow
cd xsnow

# 2. Install in editable mode with all optional dependencies
pip install -e .[gpu,dev]
```

## Core Components

The `xsnow` package is built around a clear separation of concerns: a powerful I/O function to read data, and a flexible data object to represent and analyze it.

### `xsnow.read()`: The Data Loading Function

The primary entry point for loading data is the top-level `xsnow.read()` function. It is designed to be efficient, flexible, and memory-aware.

  * **Versatile Input**: It accepts a single file path, a list of file paths, or a path to a directory.
  * **At-the-Source Filtering**: You can pass `Selector` objects (`TimeSelector`, `LocationSelector`) directly to the `read` function. This allows the parser to filter data *as it reads the files*, which can dramatically reduce memory usage by preventing unwanted data from being loaded at all.
  * **Parallel Processing**: It reads multiple files concurrently using all available CPU cores.
  * **Returns an `xsnowDataset`**: The function handles all the parsing and merging, returning a single, ready-to-use `xsnowDataset` object.

### `xsnow.xsnowDataset`: The Data Object

This is the central, user-facing object in the `xsnow` package. It acts as a smart and convenient wrapper around a standard `xarray.Dataset`. Its primary role is to hold the data and provide a high-level API for analysis and interaction.

  * **Composition over Inheritance**: This class *contains* an `xarray.Dataset` instance (in `self.data`) rather than inheriting from it. This design choice is more robust and prevents future updates to `xarray` from breaking this class.
  * **Simplified Initialization**: While the constructor accepts a pre-existing `xarray.Dataset`, the canonical way to load data from files is the `xsnow.read()` function. This makes the extension architecture clean and powerful.
  * **Seamless `xarray` Integration**: The class automatically forwards attribute and method calls (like `.sel()`, `.mean()`, `.plot`) to the underlying `xarray.Dataset`, so you can use the full power of the `xarray` API.
  * **Automatic Re-wrapping**: When you perform an operation that returns a new `xarray.Dataset`, the result is automatically wrapped back into an `xsnowDataset` instance, allowing you to chain operations together.

#### Dictionary-Style Access

To make the `xsnowDataset` feel even more like a native `xarray.Dataset`, it supports dictionary-style access for getting and setting data variables.

  * **Getting Data**: You can access any data variable using square brackets `[]`.
    ```python
    # Access the 'density' DataArray
    density_data = ds['density']
    ```
  * **Setting Data**: You can add a new variable or overwrite an existing one.
    ```python
    # Create a new DataArray
    new_variable = xr.DataArray([1, 2, 3], dims=['layer'])
    # Assign it to the dataset
    ds['new_variable'] = new_variable
    ```

## Advanced Features

### The 5D Data Model

The `xsnow` package structures all snow profile data into a consistent, five-dimensional model.

  * **`location`**: The spatial coordinate, representing a specific station or grid point.
  * **`time`**: The temporal coordinate, representing the timestamp of a specific profile.
  * **`slope`**: Represents different slope configurations (e.g., a 30-degree, south-facing slope).
  * **`realization`**: Used for ensemble simulations, where each realization is a different model run.
  * **`layer`**: The primary vertical dimension, representing an integer index from the ground up. The physical height of each layer is stored in the `height` data variable.
  * **`z`**: A non-dimension coordinate that is **automatically calculated** by `xsnow.read()`, representing the physical depth below the snow surface in meters (`z = height - HS`). The surface is at or near `z=0`.

### Integer Location Indexing

For performance and efficiency, the `location` dimension uses simple integer indices (0, 1, 2, etc.). The human-readable station names are stored as metadata and can be easily retrieved.

  * **The Mapping:** The integer-to-name mapping is stored in the coordinate's attributes: `ds.location.attrs['mapping']`.
  * **The `.labels` Accessor:** A convenient `.labels.get()` method is provided to access this mapping directly.

<!-- end list -->

```python
import xsnow

# ds contains data for 'station_A' and 'station_B'
ds = xsnow.read('path/to/data_directory')

# Get the mapping of integer indices to station names
name_map = ds.location.labels.get()
print(name_map)
# Output: {0: 'station_A', 1: 'station_B'}

# Get the name of the first location
print(name_map[0])
# Output: 'station_A'
```

### Hardware-Aware Computation (CPU/GPU)

The `xsnow` package is designed to leverage available hardware for optimal performance. The `xsnowDataset` class provides methods to move data between the CPU and GPU, and the analysis functions are hardware-aware.

  * **`.to_gpu()`**: Moves the dataset to the GPU for accelerated processing with `CuPy`.
  * **`.to_cpu()`**: Moves the data back to the CPU for tasks like saving files.

<!-- end list -->

```python
# 1. Load data (on CPU)
ds_cpu = xsnow.read('path/to/data')

# 2. Move to GPU for fast computation
ds_gpu = ds_cpu.to_gpu()

# 3. Analysis functions automatically use the GPU
ds_results_gpu = ds_gpu.compute_critical_cut_length() 

# 4. Move results back to CPU for saving
ds_results_cpu = ds_results_gpu.to_cpu()
ds_results_cpu.to_netcdf('results.nc')
```

### Handling Weather Data at Multiple Heights

The `xsnow` package has a two-part system for working with meteorological variables measured at different heights (e.g., wind speed at 2m and 10m).

1.  **Parsing:** The `read()` function recognizes a special `_height` convention in the `[HEADER]` of SMET files. If it finds `fields = ... VW ...` and `VW_height = 10`, it will automatically create a unique variable named `VW_10m`.

2.  **Stacking:** For analyses where you need to treat these measurements as a single variable with a vertical dimension, you can use the `.stack_weather_variables()` extension method.

**Example:**

```python
# ds contains variables 'TA' (at surface) and 'TA_2m'
# We want to combine them into a single variable 'TA' with a new
# 'height_above_surface' dimension.

tidy_ta = ds.stack_weather_variables(base_name='TA')
print(tidy_ta)
```

**Output:**

```
<xarray.DataArray (location: 1, time: N, height_above_surface: 2)>
...
Coordinates:
  * location              (location) int64 0
  * time                  (time) datetime64[ns] ...
  * height_above_surface  (height_above_surface) int64 0 2
```

### Plotting and Visualization

Because `xsnowDataset` is a wrapper around `xarray.Dataset`, you can use `xarray`'s powerful and convenient built-in plotting capabilities, which are based on `matplotlib`. The automatically calculated `z` coordinate is ideal for creating intuitive profile plots.

**Example: Plotting a single snow profile's density against its physical depth**

```python
import matplotlib.pyplot as plt
import xsnow

ds = xsnow.read('path/to/station_A.pro')

# Select a single profile for a specific time
profile = ds.data.isel(location=0, time=0)

# Use xarray's .plot() method with the 'z' coordinate on the y-axis
profile['density'].plot(y='z')

plt.title("Density Profile for Station A")
plt.xlabel("Density (kg m-3)")
plt.ylabel("Depth Below Surface (m)")
plt.grid(True)
plt.show()
```

## Usage Examples

Here are some practical examples of how to use the `xsnow` package.

### 1\. Loading Data

The recommended way to load data is through the `xsnow.read()` function.

**Load all data from a directory:**

```python
import xsnow

# This will find all files, read them, and return an xsnowDataset
ds = xsnow.read('path/to/data_directory/')
print(ds)
```

**Load Data with Filtering (Memory Efficient):**

This example demonstrates the power of at-the-source filtering. The parser will only process profiles from January 2025 for 'station\_B', saving memory and time.

```python
import xsnow
from xsnow.selectors import TimeSelector, LocationSelector

# Define selectors for the desired time range and location
time_filter = TimeSelector(sel=slice('2025-01-01', '2025-01-31'))
location_filter = LocationSelector(sel='station_B')

# Pass the selectors directly to the read function
ds_filtered = xsnow.read(
    'path/to/data_directory/',
    time=time_filter,
    location=location_filter
)
print(ds_filtered)
```

### 2\. Caching and Updating Data

A common workflow is to load a large, cached dataset from a NetCDF file and then update it with the most recent data. The `.update()` and `.to_netcdf()` methods are available as registered extensions.

**Load from Cache and Update:**

```python
import xsnow

# 1. Load the existing dataset from a fast-loading NetCDF cache
ds_cache = xsnow.read('path/to/historical_data.nc')

# 2. Update the dataset with new files from a directory
ds_updated = ds_cache.update('path/to/new_data_directory/')

# 3. Save the newly updated data back to the cache for next time
ds_updated.to_netcdf('path/to/historical_data.nc')
```

## Extending `xsnow` with Custom Methods

The `xsnowDataset` class is designed to be easily extended with your own analysis and I/O functions using the plugin system.

### The Plugin System (`@xsnow.register_xsnowDataset_method`)

This method is designed to make your custom function feel like a **built-in** part of the `xsnowDataset` **object**. It's the recommended approach for creating a library of reusable tools.

  * **How it Works:** You write a standard Python function that takes an `xsnowDataset` as its first argument. By adding the `@xsnow.register_xsnowDataset_method` decorator, you register the function with the `xsnow` package. The `xsnowDataset` object automatically discovers it, allowing you to call it as if it were a built-in method.

  * **Example:**

    ```python
    # In a file like 'my_analysis_plugins.py'
    import xsnow
    import xarray as xr

    @xsnow.register_xsnowDataset_method
    def calculate_top_slab_density(ds: xsnow.xsnowDataset, depth_cm: float = 20.0) -> xr.DataArray:
        """Calculates the mean density of the top slab of the snowpack."""
        # First, ensure HS is calculated
        if 'HS' not in ds.data:
            ds = ds.compute_hs()
        
        # Define the height of the bottom of the slab
        slab_bottom_height = ds.data['HS'] - (depth_cm / 100.0)
        
        # Select layers within the top slab using the 'height' data variable
        top_slab = ds.data.where(ds.data.height >= slab_bottom_height, drop=True)
        return top_slab['density'].mean(dim='layer')

    # --- In your main script or notebook ---
    import xsnow
    import my_analysis_plugins # This import registers the function

    ds = xsnow.read('path/to/data')
    # Call your function like a built-in method!
    mean_density = ds.calculate_top_slab_density(depth_cm=30)
    print(mean_density)
    ```

## Advanced Analysis Workflow Example 1: Calculating Stability and Filtering Layers

This example demonstrates how to chain together registered methods to derive meaningful insights from the data. Because the analysis functions are registered as extensions, they can be called directly on any `xsnowDataset` object.

```python
import xsnow

# 1. Load the initial data
ds = xsnow.read('path/to/your/data')

# 2. (Optional) Move to GPU for faster computation
ds_gpu = ds.to_gpu()

# 3. Compute total snow height using the built-in method
ds_with_hs = ds_gpu.compute_hs()

# 4. Compute critical cut length using the registered extension method
ds_with_rc = ds_with_hs.compute_critical_cut_length()

# 5. Define weak layer criteria and find matching layers
criteria = "density < 200 & grain_size > 0.4"
ds_filtered = ds_with_rc.find_layer_by_criteria(criteria)

# 6. Move the final result back to the CPU to print or plot
ds_final_cpu = ds_filtered.to_cpu()

# 7. Print a slice of the final dataset to inspect the results
print("Results for the first location and first timestamp:")
print(ds_final_cpu.data.isel(location=0, time=0))
```

## Advanced Analysis Workflow Example 2: Wet Slab Stability
This example demonstrates a more complex workflow for assessing a wet slab avalanche problem. It uses the registered .get_wet_slab_LOC() method to find a potential weak layer and then calculates the distance to the percolating wetting front.

```python
import xsnow
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from pathlib import Path

# --- 1. Load Data ---
# This workflow requires a .pro file with 'height', 'grain_type',
# 'grain_size', and 'lwc' variables.
pro_file_path = "path/to/your/data.pro"
ds = xsnow.read(pro_file_path)

# --- 2. Find the Layer of Concern (LOC) ---
# Use the custom registered method to find the LOC for each timestamp.
# The method returns a DataArray of the LOC's height over time.
loc_height = ds.get_wet_slab_LOC()

# --- 3. Find the Wetting Front ---
# The wetting front is the deepest layer with melt forms and significant water.
is_melt_form = (ds.data['grain_type'] >= 770) & (ds.data['grain_type'] <= 779)
is_wet = ds.data['lwc'] > 0.03
wet_front_candidates = ds.data['height'].where(is_melt_form & is_wet)
wet_front_height = wet_front_candidates.min(dim='layer')

# --- 4. Calculate Distance and Print Summary ---
distance = wet_front_height - loc_height
# Select the last timestamp for a summary report
last_time_step = ds.time.isel(time=-1)
final_distance = distance.sel(time=last_time_step, method="nearest").item()

print(f"--- Wet Slab Analysis for {pd.to_datetime(last_time_step.values).date()} ---")
if not np.isnan(final_distance):
    print(f"Distance to Wet Front: {final_distance:.2f} m")
    if final_distance < 0:
        print("WARNING: The wetting front has passed the layer of concern.")
else:
    print("A Layer of Concern or Wet Front was not found.")

# --- 5. Plot the Time Series ---
fig, ax = plt.subplots(figsize=(10, 6))
hs = ds.compute_hs().data.HS

ax.plot(ds.time.squeeze(), hs.squeeze(), color='blue', linestyle='--', label='Snow Surface (HS)')
ax.plot(ds.time.squeeze(), loc_height.squeeze(), color='black', marker='.', markersize=4, label='Layer of Concern (LOC)')
ax.plot(ds.time.squeeze(), wet_front_height.squeeze(), color='red', linestyle=':', marker='x', markersize=4, label='Wet Front')

ax.set_title('Wet Slab Analysis Time Series')
ax.set_ylabel('Height Above Ground (m)')
ax.set_xlabel('Date')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
fig.autofmt_xdate()
plt.show()
```