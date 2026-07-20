# examples/example_wet_slab_timeseries.py
"""
Example script for a wet slab stability analysis over a time period.

This workflow reads a .pro file, identifies a potential "layer of concern" (LOC)
and a wetting front for each timestamp, and visualizes their evolution over time.
"""

import sys
from pathlib import Path
import xsnow
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    """Main function to run the analysis workflow."""
    print("--- Running Wet Slab Time-Series Analysis Example ---")
    
    pro_file_path = "data/102474_res.pro" #"path/to/your/data.pro"
    
    pro_file = Path(pro_file_path)
    if not pro_file.is_file():
        print(f"Error: The specified file does not exist: '{pro_file_path}'")
        print("Please update the 'pro_file_path' variable in this script.")
        sys.exit(1)

    print(f"\n1. Reading data from '{pro_file_path}'...")
    ds = xsnow.read(pro_file)
    
    if ds is None:
        print("No data was loaded. Exiting.")
        sys.exit(1)

    # --- 2. Check for required variables ---
    required_vars = ['height', 'grain_type', 'grain_size', 'lwc']
    for var in required_vars:
        if var not in ds.data:
            print(f"Error: The loaded dataset is missing the required variable '{var}'.")
            sys.exit(1)
            
    # --- 3. Run Analysis ---
    print("\n2. Finding Layer of Concern (LOC) and Wet Front for all timestamps...")
    
    # Use the registered method, which returns a DataArray of heights over time
    loc_height = ds.get_wet_slab_LOC()

    # Find the wet front for all time steps
    is_melt_form = (ds.data['grain_type'] >= 770) & (ds.data['grain_type'] <= 779)
    is_wet = ds.data['lwc'] > 0.03
    wet_front_candidates = ds.data['height'].where(is_melt_form & is_wet)
    wet_front_height = wet_front_candidates.min(dim='layer')

    # Get total snow depth (HS) for plotting
    hs = ds.compute_hs().data.HS

    # --- 4. Plot the Results ---
    print("\n3. Generating time-series plot...")
    
    # FIX: Use .squeeze() to remove single-entry dimensions for plotting
    time_coords = ds.time.squeeze()
    hs_to_plot = hs.squeeze()
    loc_to_plot = loc_height.squeeze()
    wet_front_to_plot = wet_front_height.squeeze()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot HS, LOC height, and Wet Front height over time
    ax.plot(time_coords, hs_to_plot, color='blue', linestyle='--', label='Snow Surface (HS)')
    ax.plot(time_coords, loc_to_plot, color='black', linestyle='-', marker='.', markersize=4, label='Layer of Concern (LOC)')
    ax.plot(time_coords, wet_front_to_plot, color='red', linestyle=':', marker='x', markersize=4, label='Wet Front')

    # Formatting
    ax.set_title('Wet Slab Analysis Time Series')
    ax.set_ylabel('Height Above Ground (m)')
    ax.set_xlabel('Date')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Improve date formatting on the x-axis
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()