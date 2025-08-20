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
* **Simplified Initialization**: The constructor is designed to accept a pre-existing `xarray.Dataset`. This makes the extension architecture clean and powerful, as any function that produces an `xarray.Dataset` can be used to create a new `xsnowDataset`.
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
    new_variable = xr.DataArray([1, 2, 3], dims=['z'])
    # Assign it to the dataset
    ds['new_variable'] = new_variable
    ```

## Advanced Features

### The 5D Data Model
The `xsnow` package structures all snow profile data into a consistent, five-dimensional model. Understanding these dimensions is key to effective data analysis.

* **`location`**: The spatial coordinate, representing a specific station or grid point.
* **`time`**: The temporal coordinate, representing the timestamp of a specific profile.
* **`slope`**: Represents different slope configurations (e.g., a 30-degree, south-facing slope).
* **`realization`**: Used for ensemble simulations, where each realization is a different model run.
* **`z`**: The primary vertical coordinate, representing depth below the snow surface in meters. The surface is **z=0**, and values are **negative downwards**. The original height above ground is still available as the `height` data variable.

### Hardware-Aware Computation (CPU/GPU)
The `xsnow` package is designed to leverage available hardware for optimal performance. The `xsnowDataset` class provides methods to move data between the CPU and GPU, and the functions in `analysis.py` are hardware-aware.

* **`.to_gpu()`**: Moves the dataset to the GPU for accelerated processing with `CuPy`.
* **`.to_cpu()`**: Moves the data back to the CPU for tasks like saving files.

```python
# 1. Load data (on CPU)
ds_cpu = xsnow.read('path/to/data')

# 2. Move to GPU for fast computation
ds_gpu = ds_cpu.to_gpu()

# 3. Analysis functions automatically use the GPU
# Note: analysis functions are not yet part of the package
# ds_results_gpu = xsnow.analysis.compute_rc_flat(ds_gpu) 

# 4. Move results back to CPU for saving
# ds_results_cpu = ds_results_gpu.to_cpu()
# ds_results_cpu.to_netcdf('results.nc')
```

### Plotting and Visualization
Because `xsnowDataset` is a wrapper around `xarray.Dataset`, you can use `xarray`'s powerful and convenient built-in plotting capabilities, which are based on `matplotlib`.

**Example: Plotting a single snow profile's density**
```python
import matplotlib.pyplot as plt
import xsnow

ds = xsnow.read('path/to/station_A.pro')

# Select a single profile for a specific time
profile = ds.data.isel(location=0, time=0)

# Use xarray's .plot() method with the new 'z' coordinate
profile['density'].plot(y='z')

plt.title("Density Profile for Station A")
plt.xlabel("Density (kg m-3)")
plt.ylabel("Depth Below Surface (m)")
plt.grid(True)
plt.show()
```

## Usage Examples

Here are some practical examples of how to use the `xsnow` package.

### 1. Loading Data

The recommended way to load data is through the `xsnow.read()` function.

**Load all data from a directory:**
```python
import xsnow

# This will find all files, read them, and return an xsnowDataset
ds = xsnow.read('path/to/data_directory/')
print(ds)
```

**Load Data with Filtering (Memory Efficient):**

This example demonstrates the power of at-the-source filtering. The parser will only process profiles from January 2025 for 'station_B', saving memory and time.
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

### 2. Merging SMET Data

You can add meteorological data from a `.smet` file to an existing `xsnowDataset` object.
```python
import xsnow

# 1. Load your primary snow profile data
ds_profiles = xsnow.read('path/to/station_A.pro')

# 2. Merge the corresponding SMET data
ds_merged = ds_profiles.merge_smet('path/to/station_A.smet')

print(list(ds_merged.data.data_vars)) # Will now include TA, VW, etc.
```

### 3. Caching and Updating Data

A common workflow is to load a large, cached dataset from a NetCDF file and then update it with the most recent data.

**Load from Cache and Update:**
```python
import xsnow

# 1. Load the existing dataset from a fast-loading NetCDF cache
# Note: we still use xsnow.read() for all file types
ds_cache = xsnow.read('path/to/historical_data.nc')

# 2. Update the dataset with new files
ds_updated = ds_cache.update('path/to/new_data_directory/')

# 3. Save the newly updated data back to the cache for next time
ds_updated.to_netcdf('path/to/historical_data.nc')
```

### 4. Saving and Exporting Data

You can easily save your processed data to various formats.

**Save to NetCDF (Recommended for Caching):**
```python
import xsnow

ds = xsnow.read('path/to/data_directory/')
# ... perform some analysis ...
processed_ds = ds.compute_hs()

# Save the processed data for fast reloading later
processed_ds.to_netcdf('processed_cache.nc')
```

## Extending `xsnow` with Custom Functions

The `xsnowDataset` class is designed to be easily extended with your own analysis functions. There are two primary patterns for this, each suited to different needs.

### Method 1: The Plugin System (`@xsnow.register_analysis`)

This method is designed to make your custom function feel like a **built-in** part of the `xsnowDataset` **object**. It's the recommended approach for creating a library of reusable tools.

* **How it Works:** You write a standard Python function that takes an `xsnowDataset` as its first argument. By adding the `@xsnow.register_analysis` decorator above it, you "sign up" your function with the `xsnow` package. The `xsnowDataset` object automatically discovers this function, allowing you to call it as if it were a built-in method.

* **Example:**
  ```python
  # In a file like 'my_analysis_plugins.py'
  import xsnow
  import xarray as xr
  
  @xsnow.register_analysis
  def calculate_top_slab_density(ds: xsnow.xsnowDataset, depth_cm: float = 20.0) -> xr.DataArray:
      """Calculates the mean density of the top slab of the snowpack."""
      # Select layers within the top slab using the z-coordinate
      top_slab = ds.data.where(ds.data.z >= -depth_cm / 100.0, drop=True)
      return top_slab['density'].mean(dim='z')
  
  # --- In your main script or notebook ---
  import xsnow
  import my_analysis_plugins # This import registers the function
  
  ds = xsnow.read('path/to/data')
  # Call your function like a built-in method!
  mean_density = ds.calculate_top_slab_density(depth_cm=30)
  print(mean_density)
  ```

### Method 2: The Standalone Function

This approach treats the analysis as an **external operation** that you perform *on* the dataset. It's simpler and more direct, making it ideal for analyses that are specific to a single script or project.

* **How it Works:** You write a standard Python function that accepts an `xsnowDataset` object as an argument. The function performs its calculations on the internal `xarray.Dataset` (`ds.data`) and then, at the very end, it creates and returns a **new** `xsnowDataset` instance containing the results.

* **Example:**
  ```python
  import xsnow
  
  def add_slab_density_variable(ds: xsnow.xsnowDataset, slab_thickness: float = 0.3) -> xsnow.xsnowDataset:
      """
      Calculates the average density of the top slab and adds it as a new
      variable to the dataset.
      """
      # Select layers in the slab using the z-coordinate
      slab_layers = ds.data['density'].where(ds.data.z >= -slab_thickness)
      slab_density = slab_layers.mean(dim='z')
  
      new_xarray_ds = ds.data.assign(slab_density=slab_density)
  
      # The key step: re-wrap the modified xarray.Dataset in our class
      return xsnow.xsnowDataset(new_xarray_ds)
  
  # --- Usage ---
  ds = xsnow.read('path/to/data')
  extended_ds = add_slab_density_variable(ds)
  print(extended_ds.data['slab_density'])
  ```

### Which Method Should You Use?

| Use the... | When you want... | 
| :--- | :--- |
| **Plugin System (`@register_analysis`)** | ...to create a **reusable library** of functions that feel like built-in methods. Prioritize **convenience and a clean, chainable API** for the end-user (e.g., `ds.compute_hs().calculate_top_slab_density()`). | 
| **Standalone Function** | ...to perform a **specific analysis** for a single script or project. Prioritize **explicitness and a clear, functional data flow** (`in -> process -> out`). | 

## Advanced Analysis Workflow Example

This example demonstrates how to chain together analysis functions to derive a more complex metric: the average density of the snow slab resting above a potential weak layer.

First, we define an extension function using the "Standalone Function" pattern. This function will use tools from the `analysis` module.

```python
import xsnow
# from xsnow.analysis import find_layer_by_criteria # Assuming this will be available

def calculate_mean_density_above_weak_layer(ds: xsnow.xsnowDataset, criteria: str) -> xsnow.xsnowDataset:
    """
    Calculates the mean density of the snowpack slab above the highest
    layer that meets the specified weak layer criteria.
    """
    # 1. Find all layers that match the criteria (conceptual)
    # weak_layers = find_layer_by_criteria(ds, criteria)
    
    # 2. Find the depth of the highest weak layer (conceptual)
    # highest_weak_layer_z = weak_layers.data['z'].max(dim='z')
    
    # 3. Create a mask for all layers *above* the highest weak layer (conceptual)
    # slab_mask = ds.data.z > highest_weak_layer_z
    
    # 4. Calculate the mean density of those layers (conceptual)
    # mean_density_above_wl = ds.data['density'].where(slab_mask).mean(dim='z')
    
    # For the example, create a dummy result
    mean_density_above_wl = ds.data['density'].mean(dim='z')
    
    mean_density_above_wl.attrs = {
        'long_name': f'Mean density above highest weak layer ({criteria})',
        'units': 'kg m-3'
    }

    # 5. Add the new variable to the dataset and re-wrap
    new_xarray_ds = ds.data.assign(mean_density_above_wl=mean_density_above_wl)
    return xsnow.xsnowDataset(new_xarray_ds)
```

Now, we can use this function in a complete analysis workflow that demonstrates the new API.

```python
import xsnow
# from xsnow.analysis import compute_rc_flat, find_layer_by_criteria

# 1. Load the initial data using the top-level read function
ds = xsnow.read('path/to/your/data')

# 2. (Optional) Move to GPU for faster computation
ds_gpu = ds.to_gpu()

# 3. Compute a built-in analysis variable (conceptual)
# ds_with_rc = compute_rc_flat(ds_gpu)
ds_with_rc = ds_gpu # Placeholder

# 4. Define weak layer criteria
criteria = "density < 200"

# 5. Use our standalone extension function to calculate the slab density
ds_final = calculate_mean_density_above_weak_layer(ds_with_rc, criteria)

# 6. Move the final result back to the CPU to print or plot
ds_final_cpu = ds_final.to_cpu()
print(ds_final_cpu.data['mean_density_above_wl'])