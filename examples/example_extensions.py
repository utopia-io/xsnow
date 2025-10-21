# example_extensions.py
"""
Usage examples for the xsnow package, demonstrating the two primary
extension patterns for adding new functionality.

This script shows how to:
1.  Load data using the `read()` function.
2.  Apply a "Simple Extension" by using a standalone function to add a
    new data variable (e.g., slab density).
3.  Apply a "Complex Extension" by using a decorator class to add a
    new data stream with its own dimension (e.g., hazard chart data).

---
Setup and Execution
---
To run this script, you must first install the `xsnow` package in
"editable" mode from your project's root directory:

    uv pip install -e .
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

from xsnow import read
from xsnow.core import xsnowDataset, DatasetDecorator


# --- Example 1: Simple Extension (Standalone Function) ---

def add_slab_density(ds: xsnowDataset, slab_thickness: float = 0.3) -> xsnowDataset:
    """
    Calculates the average density of the top slab of the snowpack.
    This is an example of a simple extension that adds a new variable.
    
    Args:
        ds: The input xsnowDataset
        slab_thickness: Thickness of the top slab in meters (default: 0.3m)
    
    Returns:
        A new xsnowDataset with the 'slab_density' variable added
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
    
    new_xarray_ds = ds.data.assign(slab_density=slab_density)
    
    # The key step: re-wrap the modified xarray.Dataset in our class
    return xsnowDataset(new_xarray_ds)


# --- Example 2: Complex Extension (Decorator Class) ---

class HazardChartData(DatasetDecorator):
    """
    A decorator that adds a new 'hazardChartElement' dimension and data.
    This is an example of a complex extension that adds new dimensions.
    
    This pattern is useful when you want to attach a completely new
    data stream (like hazard assessment scores) that has its own
    dimension structure separate from the snowpack layers.
    """
    def __init__(self, base: xsnowDataset):
        """
        Initialize the decorator with hazard chart data.
        
        Args:
            base: The base xsnowDataset to decorate
        """
        super().__init__(base)
        # Only compute the new data if it's not already present.
        if 'hazard_value' not in self.data.data_vars:
            self.data = self._compute_hazard_elements()

    def _compute_hazard_elements(self):
        """
        Computes the hazard elements. This method is robust to subsetting.
        
        In a real application, this would contain actual hazard assessment
        logic. Here we just generate random values for demonstration.
        
        Returns:
            xarray.Dataset with hazard_value variable added
        """
        num_elements = 5
        base_dims = [dim for dim in ['location', 'time'] if dim in self.data.dims]
        shape = [self.data.sizes[dim] for dim in base_dims] + [num_elements]
        dims = base_dims + ['hazardChartElement']

        hazard_values = xr.DataArray(
            np.random.rand(*shape),
            dims=dims
        ).assign_coords(hazardChartElement=np.arange(num_elements))
        
        hazard_values.attrs = {
            'long_name': 'Hazard assessment scores',
            'units': 'dimensionless',
            'description': 'Random hazard scores for demonstration (0-1 scale)'
        }
        
        return self.data.merge(hazard_values.rename('hazard_value'))


# --- Helper Function for Clean Output ---

def print_dataset(title, dataset):
    """Pretty print a dataset with a title."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(dataset)
    print("="*60)


# --- Main Execution Block ---

def main():
    """
    Main function to run the examples.
    """
    # --- Setup: Create dummy data files ---
    print("\n" + "="*60)
    print("XSNOW EXTENSIONS EXAMPLE")
    print("="*60)
    
    print("\n--- 1. Setting up dummy data files ---")
    dummy_files = []
    
    for i in range(2):
        # Create synthetic snow profile data
        heights = np.linspace(0.0, 1.0, 10)
        densities = np.full(10, 250.0 + i * 50)  # Vary density by station
        
        content = f"""[STATION_PARAMETERS]
StationName= DUMMY_STATION_{i}
Latitude= {46.8 + i*0.1}
Longitude= {9.8 + i*0.1}
Altitude= {2540 + i*10}

[DATA]
0500,13.08.2025 12:00:00
0501,10,{",".join(map(str, heights))}
0502,10,{",".join(map(str, densities))}
"""
        dummy_filepath = f"dummy_station_{i}.pro"
        with open(dummy_filepath, "w") as f:
            f.write(content)
        dummy_files.append(dummy_filepath)
    
    print(f"Created {len(dummy_files)} dummy .pro files.")

    # --- Load Data ---
    print("\n--- 2. Loading initial data ---")
    ds = read(dummy_files)
    
    if ds is None or ds.data is None:
        print("ERROR: Failed to load data!")
        return
    
    print_dataset("Original Dataset", ds)
    
    # Show location coordinates (now strings!)
    print("\nLocation coordinates:")
    print(f"  Type: {ds.data.location.dtype}")
    print(f"  Values: {list(ds.data.location.values)}")

    # --- Apply Simple Extension ---
    print("\n--- 3. Applying Simple Extension: add_slab_density() ---")
    extended_ds = add_slab_density(ds, slab_thickness=0.5)
    
    print("\n✓ Successfully added 'slab_density' variable")
    print("\nSlab Density values:")
    print(extended_ds.data['slab_density'].values)
    
    # Show that the extension created a new variable
    print("\nVariables in extended dataset:")
    for var in extended_ds.data.data_vars:
        print(f"  - {var}")

    # --- Apply Complex Extension ---
    print("\n--- 4. Applying Complex Extension: HazardChartData Decorator ---")
    decorated_ds = HazardChartData(ds)
    
    print("\n✓ Successfully decorated dataset with hazard chart data")
    print("\nNew dimensions:")
    for dim, size in decorated_ds.data.sizes.items():
        print(f"  - {dim}: {size}")
    
    print("\nVariables in decorated dataset:")
    for var in decorated_ds.data.data_vars:
        print(f"  - {var}")
    
    print("\nHazard values shape:")
    print(f"  {decorated_ds.data['hazard_value'].shape}")

    # --- Demonstrate Subsetting with String Coordinates ---
    print("\n--- 5. Subsetting the Decorated Dataset ---")
    subset = decorated_ds.sel(location='DUMMY_STATION_0')
    
    if subset is None or subset.data is None:
        print("ERROR: Subsetting failed!")
        return
    
    print(f"\n✓ Selected location: 'DUMMY_STATION_0'")
    print(f"Result type: {type(subset).__name__}")
    print(f"Is still a HazardChartData instance: {isinstance(subset, HazardChartData)}")
    
    print("\nSubset dimensions:")
    for dim, size in subset.data.sizes.items():
        print(f"  - {dim}: {size}")
    
    # Verify the location dimension was reduced
    assert 'location' not in subset.data.sizes, "Location dimension should be removed after selection"
    print("\n✓ Dimension 'location' successfully removed after selection")
    
    # Show hazard values for the selected location
    print("\nHazard values for DUMMY_STATION_0:")
    print(subset.data['hazard_value'].values)

    # --- Demonstrate Chaining ---
    print("\n--- 6. Chaining Extensions ---")
    # Apply simple extension, then complex extension
    chained_ds = add_slab_density(ds)
    chained_ds = HazardChartData(chained_ds)
    
    print("\n✓ Successfully chained both extensions")
    print("\nFinal dataset variables:")
    for var in chained_ds.data.data_vars:
        attrs = chained_ds.data[var].attrs
        long_name = attrs.get('long_name', 'No description')
        print(f"  - {var}: {long_name}")

    # --- Cleanup ---
    print("\n--- 7. Cleaning up dummy files ---")
    for f in dummy_files:
        Path(f).unlink()
    print("✓ Cleanup complete.")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()