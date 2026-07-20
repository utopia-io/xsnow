# read_and_plot_example.py
"""
Example script demonstrating how to use the xsnow package to read, process,
and plot snowpack data from actual .pro and .smet files.

This script performs the following steps:
1.  Specifies a path to a directory containing your data files.
2.  Initializes an `xsnowDataset` object from the directory, which automatically
    reads and merges the data from both .pro and .smet files.
3.  Computes the total snow height (HS) from the profile layer data.
4.  Computes the 24-hour change in snow height (HN24).
5.  Generates and displays a plot of HS and HN24 over time.

To run this script, ensure you have the `xsnow` package and `matplotlib`
installed in your environment.
"""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from xsnow import read


def main():
    """Main function to run the example workflow."""

    # --- 1. Specify the path to your data files ---
    data_directory = r"C:\Users\Avalanche\Documents\projects\SNOWPACK\post_processing\hazard_forecasting\data\external\snowpack\2023\zone009\098441"
    
    data_dir = Path(data_directory)
    if not data_dir.is_dir():
        print(f"Error: The specified directory does not exist: '{data_directory}'")
        print("Please update the 'data_directory' variable in this script.")
        sys.exit(1)

    print(f"--- 1. Reading data from '{data_directory}' ---")

    # --- 2. Load and process the data ---
    print("\n--- 2. Loading and processing data ---")
    
    # ✨ CHANGED: Use read() function
    ds = read(str(data_dir))
    
    if ds is None or ds.data is None:
        print("No data was loaded. Exiting.")
        sys.exit(1)
        
    # Compute the total snow height (HS) and 24-hour change (HN24)
    processed_ds = ds.compute_hs().compute_hn24()
    
    print("Processed Dataset:")
    print(processed_ds.data)

    # --- 3. Plot the results ---
    print("\n--- 3. Generating plot ---")
    if processed_ds.data:
        plot_data = processed_ds.data.squeeze()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # ✨ UPDATED: Location is now a string
        station_name = str(plot_data.location.values) if 'location' in plot_data.coords else "Snowpack Analysis"
        fig.suptitle(f"Snowpack Analysis for {station_name}", fontsize=16)

        # Plot Snow Height (HS)
        plot_data['HS'].plot(ax=ax1, marker='o', color='royalblue')
        ax1.set_title("Total Snow Height (HS)")
        ax1.set_ylabel("Height (m)")
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 24-hour Snow Height Change (HN24)
        plot_data['HN24'].plot(ax=ax2, marker='o', color='seagreen')
        ax2.set_title("24-Hour Snow Height Change (HN24)")
        ax2.set_ylabel("Change (m)")
        ax2.set_xlabel("Date")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        print("Displaying plot...")
        plt.show()


if __name__ == "__main__":
    main()