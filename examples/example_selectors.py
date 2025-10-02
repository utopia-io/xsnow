# example_selectors.py
"""
Example script demonstrating how to use the Selector classes from the
xsnow.selectors module to perform structured, reusable queries.

This script performs the following steps:
1.  Creates a temporary directory with several dummy .pro files representing
    different locations and times.
2.  Loads the data into an `xsnowDataset` object.
3.  Demonstrates how to use `LocationSelector` to filter by a specific
    station ID.
4.  Demonstrates how to use `LocationSelector` to filter by a geographic
    bounding box (latitude/longitude range).
5.  Demonstrates how to use `TimeSelector` to filter by a date range.
6.  Shows how to chain selectors together to create a complex query.
7.  Cleans up the dummy files and directory.

To run this script, ensure the `xsnow` package is installed in your
environment (e.g., `uv pip install -e .`).
"""

from pathlib import Path
import shutil
import pandas as pd

# Import the main class and the selector classes
from xsnow.dataset import xsnowDataset
from xsnow.selectors import LocationSelector, TimeSelector


def main():
    """Main function to run the selector examples."""

    # --- 1. Setup: Create dummy data files ---
    print("--- 1. Setting up dummy data files ---")
    data_dir = Path("selector_example_data")
    data_dir.mkdir(exist_ok=True)

    # Create dummy .pro files for three different stations
    station_data = {
        'station_A': {'lat': 45.0, 'lon': 10.0},
        'station_B': {'lat': 45.5, 'lon': 10.5},
        'station_C': {'lat': 46.0, 'lon': 11.0}
    }

    for name, coords in station_data.items():
        content = f"""
[STATION_PARAMETERS]
StationName= {name}
Latitude= {coords['lat']}
Longitude= {coords['lon']}
Altitude= 2000
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0500,02.01.2025 12:00:00
0501,2,0.15,0.25
0500,03.01.2025 12:00:00
0501,2,0.2,0.3
"""
        (data_dir / f"{name}.pro").write_text(content)
    print(f"Created {len(station_data)} dummy .pro files in '{data_dir}/'")

    # --- 2. Load initial data ---
    print("\n--- 2. Loading initial data ---")
    ds = xsnowDataset(str(data_dir))
    print("Original Dataset:")
    print(ds)

    # --- 3. Use LocationSelector by ID ---
    print("\n--- 3. Selecting a single location by ID ('station_B') ---")
    selector_id = LocationSelector(sel='station_B')
    subset_by_id = xsnowDataset(selector_id.apply(ds.data))
    print("Result:")
    print(subset_by_id)

    # --- 4. Use LocationSelector by Bounding Box ---
    print("\n--- 4. Selecting locations within a geographic bounding box ---")
    # This bounding box should select station_A and station_B
    selector_bbox = LocationSelector(lat_range=(44.9, 45.6), lon_range=(9.9, 10.6))
    subset_by_bbox = xsnowDataset(selector_bbox.apply(ds.data))
    print("Result (should contain station_A and station_B):")
    print(subset_by_bbox)

    # --- 5. Use TimeSelector ---
    print("\n--- 5. Selecting a time slice ('2025-01-02' to '2025-01-03') ---")
    selector_time = TimeSelector(sel=slice('2025-01-02', '2025-01-03'))
    subset_by_time = xsnowDataset(selector_time.apply(ds.data))
    print("Result:")
    print(subset_by_time)

    # --- 6. Chain Selectors for a Complex Query ---
    print("\n--- 6. Chaining selectors: BBox and Time ---")
    # Start with the full dataset's xarray object
    complex_subset_xr = ds.data
    # Apply the first selector
    complex_subset_xr = selector_bbox.apply(complex_subset_xr)
    # Apply the second selector to the result of the first
    complex_subset_xr = selector_time.apply(complex_subset_xr)
    # Wrap the final result
    complex_subset = xsnowDataset(complex_subset_xr)
    print("Result (stations A & B, for Jan 2nd and 3rd):")
    print(complex_subset)

    # --- 7. Cleanup ---
    print("\n--- 7. Cleaning up dummy files ---")
    shutil.rmtree(data_dir)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
