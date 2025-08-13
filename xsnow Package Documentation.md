xsnow Package Documentation
This document provides an overview of the core components of the xsnow package, a powerful tool for reading, analyzing, and post-processing SNOWPACK model output. It details the data I/O module (xsnow_io.py) and the main data object (xsnowDataset), along with practical examples of how to use them.

Core Components
The xsnow package is built around two primary modules that handle data ingestion and representation.

xsnow_io.py: Data Input/Output
This module is the engine for reading SNOWPACK .pro files. It is designed to be efficient and flexible, handling everything from a single file to large collections.

read(source, parallel=True): This is the main, high-level function for loading data.

Versatile Input: It accepts a single file path, a list of file paths, or a path to a directory containing .pro files.

Parallel Processing: By default, it reads multiple files concurrently using all available CPU cores, which dramatically speeds up the loading process for large datasets. 🚀

Automatic Merging: It intelligently combines the data from all files into a single, consolidated xarray.Dataset, correctly aligning them along the location dimension.

read_pro(filepath, ...): This is the low-level helper function that parses a single .pro file. It's called by read() for each file and is responsible for translating the raw text data into the structured, multi-dimensional format used by the package.

dataset.py: The xsnowDataset Class
This is the central, user-facing object in the xsnow package. It acts as a smart and convenient wrapper around a standard xarray.Dataset.

Flexible Initialization: You can create an xsnowDataset object from a file path, a list of files, a directory, or even an existing xarray.Dataset. This flexibility is the key to the package's extension architecture.

Seamless xarray Integration: The class automatically forwards attribute and method calls (like .sel(), .mean(), .plot, or .sizes) to the underlying xarray.Dataset. This means you can interact with it just like you would with a normal xarray object.

Automatic Re-wrapping: When you perform an operation that returns a new xarray.Dataset (e.g., subsetting the data), the result is automatically wrapped back into an xsnowDataset instance. This ensures you can chain operations together without losing the convenience of the wrapper class.

Usage Examples
Here are some practical examples of how to use the xsnow package.

1. Loading Data
You can load data in several ways, depending on how your files are organized.

Load a single .pro file:

import xsnow

# The xsnowDataset class calls the read() function internally
ds = xsnow.xsnowDataset('path/to/your/station.pro')
print(ds)

Load a list of specific .pro files:

import xsnow

file_list = ['path/to/station_1.pro', 'path/to/station_2.pro']
ds = xsnow.xsnowDataset(file_list)
print(ds)

Load all .pro files from a directory (in parallel):

import xsnow

# This will find all .pro files in the directory and read them concurrently
ds = xsnow.xsnowDataset('path/to/pro_files_directory/')
print(ds)

2. Accessing and Subsetting Data
Because xsnowDataset forwards calls to xarray, you can use all the standard xarray methods for data selection and analysis.

# Load some data
ds = xsnow.xsnowDataset('path/to/pro_files_directory/')

# Select data for a specific location
station_wf_sierra = ds.sel(location='WF_Sierra')

# Select a specific time slice from that location's data
january_data = station_wf_sierra.sel(time=slice('2025-01-01', '2025-01-31'))

# The result of these operations is still an xsnowDataset object
print(type(january_data))
# Output: <class 'xsnow.dataset.xsnowDataset'>

# You can access dimensions and coordinates as you would in xarray
print(january_data.sizes)

3. Saving Data
You can easily save your processed data to a NetCDF file for efficient reloading later or for use in other applications.

# Load and process some data
ds = xsnow.xsnowDataset('path/to/pro_files_directory/')
subset = ds.sel(time='2025-02-10')

# Save the subset to a new file
subset.to_netcdf('february_10_data.nc')
print("Data saved successfully!")

This documentation provides a starting point for using and understanding the core features of the xsnow package.