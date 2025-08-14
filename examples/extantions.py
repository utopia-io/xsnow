# examples.py
"""
Usage examples for the xsnow package, demonstrating the two primary
extension patterns for adding new functionality.

This script shows how to:
1.  Load data using the main `xsnowDataset` class.
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

# Import the main class from the installed xsnow package
from xsnow.dataset import xsnowDataset

# --- Example 1: Simple Extension (Standalone Function) ---

def add_slab_density(ds: xsnowDataset, slab_thickness: float = 0.3) -> xsnowDataset:
    """
    Calculates the average density of the top slab of the snowpack.
    This is an example of a simple extension that adds a new variable.
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

class DatasetDecorator:
    """Generic decorator base class to handle attribute forwarding and re-wrapping."""
    def __init__(self, base: xsnowDataset):
        # The decorator should have its own reference to the data,
        # which can be modified by subclasses.
        self.data = base.data

    def __getattr__(self, name):
        # Forward attribute access to the internal xarray.Dataset
        attr = getattr(self.data, name)
        if callable(attr):
            def method_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, xr.Dataset):
                    # Re-wrap the result in a new base class, then decorate it
                    new_base = xsnowDataset(result)
                    return self.__class__(new_base)
                return result
            return method_wrapper
        return attr

class HazardChartData(DatasetDecorator):
    """
    A decorator that adds a new 'hazardChartElement' dimension and data.
    This is an example of a complex extension.
    """
    def __init__(self, base: xsnowDataset):
        super().__init__(base)
        # Only compute the new data if it's not already present.
        if 'hazard_value' not in self.data.data_vars:
            self.data = self._compute_hazard_elements()

    def _compute_hazard_elements(self):
        """
        Computes the hazard elements. This method is robust to subsetting.
        """
        num_elements = 5
        base_dims = [dim for dim in ['location', 'time'] if dim in self.data.dims]
        shape = [self.data.sizes[dim] for dim in base_dims] + [num_elements]
        dims = base_dims + ['hazardChartElement']

        hazard_values = xr.DataArray(
            np.random.rand(*shape),
            dims=dims
        ).assign_coords(hazardChartElement=np.arange(num_elements))
        
        return self.data.merge(hazard_values.rename('hazard_value'))


# --- Main Execution Block ---

def main():
    """
    Main function to run the examples.
    """
    # --- Setup: Create dummy data files ---
    print("--- 1. Setting up dummy data files ---")
    dummy_files = []
    for i in range(2):
        content = f"""
[STATION_PARAMETERS]
StationName= DUMMY_STATION_{i}
Latitude= {46.8 + i*0.1}
Longitude= {9.8 + i*0.1}
Altitude= {2540 + i*10}
[DATA]
0500,13.08.2025 12:00:00
0501,10,{",".join(map(str, np.linspace(0.0, 1.0, 10)))}
0502,10,{",".join(map(str, np.full(10, 250.0)))}
"""
        dummy_filepath = f"dummy_station_{i}.pro"
        with open(dummy_filepath, "w") as f:
            f.write(content)
        dummy_files.append(dummy_filepath)
    print(f"Created {len(dummy_files)} dummy .pro files.")

    # --- Load initial data ---
    print("\n--- 2. Loading initial data ---")
    ds = xsnowDataset(dummy_files)
    print("Original Dataset:")
    print(ds)

    # --- Apply Simple Extension ---
    print("\n--- 3. Applying Simple Extension: add_slab_density() ---")
    extended_ds = add_slab_density(ds)
    print("Extended Dataset with 'slab_density':")
    print(extended_ds)
    print("\nSlab Density values:")
    print(extended_ds.data['slab_density']) # pyright: ignore[reportOptionalSubscript]

    # --- Apply Complex Extension ---
    print("\n--- 4. Applying Complex Extension: HazardChartData Decorator ---")
    decorated_ds = HazardChartData(ds)
    print("Decorated Dataset with 'hazard_value' and 'hazardChartElement' dimension:")
    print(decorated_ds.data)

    # --- Demonstrate that the decorated object is still fully functional ---
    print("\n--- 5. Subsetting the Decorated Dataset ---")
    subset = decorated_ds.sel(location='DUMMY_STATION_0')
    print("Subset of decorated data:")
    print(subset.data)
    assert isinstance(subset, HazardChartData)
    assert 'location' not in subset.sizes

    # --- Cleanup ---
    print("\n--- 6. Cleaning up dummy files ---")
    for f in dummy_files:
        Path(f).unlink()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
