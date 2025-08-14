# example_analysis.py
"""
Example script demonstrating how to use the analysis functions from the
xsnow package.

This script shows a complete workflow:
1.  Loads snow profile data from a directory.
2.  Optionally moves the data to a GPU for accelerated processing.
3.  Applies the `compute_rc_flat` function to calculate ram resistance.
4.  Applies the `find_layer_by_criteria` function to identify weak layers.
5.  Uses an extension function to calculate the mean density of the slab
    above the highest weak layer.
6.  Prints a comparison of the results.

To run this script:
1.  Make sure the `xsnow` package is installed (`uv pip install -e .`).
2.  Update the `data_directory` variable to point to your data.
3.  Run from the terminal: `python path/to/example_analysis.py`
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Import the main dataset class and the analysis functions
from xsnow.dataset import xsnowDataset
from xsnow.analysis import compute_critical_cut_length, find_layer_by_criteria


def calculate_mean_density_above_weak_layer(ds: xsnowDataset, criteria: str) -> xsnowDataset:
    """
    Calculates the mean density of the snowpack slab above the highest
    layer that meets the specified weak layer criteria.

    This is an example of a simple extension that combines existing analysis
    tools to create a new derived variable. It is hardware-aware.

    Args:
        ds (xsnowDataset): The input dataset.
        criteria (str): The criteria to identify weak layers.

    Returns:
        xsnowDataset: A new dataset with the 'mean_density_above_wl' variable added.
    """
    if ds.data is None:
        raise ValueError("Input dataset is empty.")

    # --- Hardware-Aware Computation Setup ---
    xp = np
    if hasattr(ds.data.variables[list(ds.data.data_vars)[0]].data, 'device'):
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            pass # Fallback to numpy if cupy isn't available

    # 1. Find all layers that match the criteria
    weak_layers = find_layer_by_criteria(ds, criteria)

    # 2. Find the depth of the highest weak layer in each profile
    highest_weak_layer_depth = weak_layers.data['height'].max(dim='layer')

    # 3. Create a mask for all layers that are *above* the highest weak layer
    # FIX: Perform the comparison on the raw arrays using the hardware-aware backend (xp)
    # This avoids a TypeError when broadcasting a GPU array against another.
    height_data = xp.asarray(ds.data['height'].data)
    # Add a new axis to the depth array so it can be broadcast correctly against the 3D height array
    highest_depth_data = xp.asarray(highest_weak_layer_depth.data)[..., xp.newaxis]
    slab_mask_data = height_data > highest_depth_data
    slab_mask = xr.DataArray(slab_mask_data, dims=ds.data['height'].dims, coords=ds.data['height'].coords)


    # 4. Calculate the mean density using the hardware-aware backend
    density_data = ds.data['density'].data
    # Apply the mask using the hardware-aware xp.where
    masked_density_data = xp.where(slab_mask.data, density_data, xp.nan)
    
    # Use nanmean along the layer axis (which is the last axis, -1)
    mean_density_data = xp.nanmean(masked_density_data, axis=-1)

    # Reconstruct the DataArray with the correct dimensions and coordinates
    result_dims = [dim for dim in ds.data['density'].dims if dim != 'layer']
    result_coords = {dim: ds.data[dim] for dim in result_dims}
    
    mean_density_above_wl = xr.DataArray(
        mean_density_data,
        dims=result_dims,
        coords=result_coords
    )
    
    mean_density_above_wl.attrs = {
        'long_name': f'Mean density above highest weak layer ({criteria})',
        'units': 'kg m-3'
    }

    # 5. Add the new variable to the dataset
    new_xarray_ds = ds.data.assign(mean_density_above_wl=mean_density_above_wl)
    
    return xsnowDataset(new_xarray_ds)


def main():
    """Main function to run the analysis workflow."""

    # --- 1. Specify the path to your data files ---
    # IMPORTANT: Change this path to the directory containing your .pro files.
    data_directory = "path/to/your/data"
    
    data_dir = Path(data_directory)
    if not data_dir.is_dir():
        print(f"Error: The specified directory does not exist: '{data_directory}'")
        print("Please update the 'data_directory' variable in this script.")
        sys.exit(1)

    print(f"--- 1. Reading data from '{data_directory}' ---")
    ds_cpu = xsnowDataset(str(data_dir))

    if ds_cpu.data is None:
        print("No data was loaded. Exiting.")
        sys.exit(1)

    print("\nOriginal Dataset:")
    print(ds_cpu)

    # --- 2. (Optional) Move data to GPU ---
    print("\n--- 2. Attempting to move data to GPU ---")
    ds_gpu = ds_cpu.to_gpu()

    # --- 3. Compute Critical Cut Length ---
    print("\n--- 3. Computing Critical Cut Length (rc_flat) ---")
    # The compute function is hardware-aware and will use the GPU if available
    ds_with_rc = compute_critical_cut_length(ds_gpu)
    
    # Move data back to CPU for printing
    ds_with_rc_cpu = ds_with_rc.to_cpu()
    
    # Select the middle timestamp for a more representative example
    middle_time_index = ds_with_rc_cpu.data.sizes['time'] // 2
    middle_timestamp = ds_with_rc_cpu.data.time.values[middle_time_index]
    
    print(f"rc_flat for profile at {pd.to_datetime(middle_timestamp).date()}:")
    print(ds_with_rc_cpu.data['rc_flat'].isel(location=0, time=middle_time_index).values.round(2))

    # --- 4. Find Weak Layers ---
    print("\n--- 4. Finding weak layers with criteria ---")
    criteria = "density < 200"
    print(f"Criteria: {criteria}")
    
    # The filtering function is also hardware-aware
    weak_layers_ds = find_layer_by_criteria(ds_with_rc, criteria)
    
    # Move data back to CPU for printing
    weak_layers_cpu = weak_layers_ds.to_cpu()

    # --- 5. Calculate Mean Density Above Weak Layer ---
    print("\n--- 5. Calculating mean density above the weak layer ---")
    ds_final = calculate_mean_density_above_weak_layer(ds_gpu, criteria) # Use GPU version
    
    # Move back to CPU for printing
    ds_final_cpu = ds_final.to_cpu()
    
    print("Mean density above weak layer for first location:")
    # Use .values to get the numpy array for clean printing
    print(ds_final_cpu.data['mean_density_above_wl'].isel(location=0).values.round(2))

    # --- 6. Compare and Print Results ---
    print(f"\n--- 6. Comparing original vs. filtered profile for {pd.to_datetime(middle_timestamp).date()} ---")
    # Squeeze to make it easier to view a single profile
    original_profile = ds_cpu.data.isel(location=0, time=middle_time_index)
    filtered_profile = weak_layers_cpu.data.isel(location=0, time=middle_time_index)
    
    print("\nOriginal Density Profile:")
    print(original_profile['density'].values.round(2))
    
    print("\nFiltered Density Profile (non-matching layers are NaN):")
    print(filtered_profile['density'].values.round(2))
    
    print("\nOriginal Grain Size Profile:")
    print(original_profile['grain_size'].values.round(2))
    
    print("\nFiltered Grain Size Profile (non-matching layers are NaN):")
    print(filtered_profile['grain_size'].values.round(2))


if __name__ == "__main__":
    main()
