# example_customanalysis_plugin.py
"""
This script demonstrates how to use the custom analysis plugin.

It performs the following steps:
1. Creates a dummy SNOWPACK .pro file for demonstration purposes.
2. Adds the 'src' directory to the path so the xsnow package can be found.
3. Imports our custom plugin from `example_plugin_analysis.py`. This step is crucial
   as it runs the decorator and registers our function.
4. Creates an `xsnowDataset` instance from the dummy data file.
5. Calls the new `.calculate_top_slab_density()` method on the dataset.
6. Prints the results.
"""
import sys
from pathlib import Path
import numpy as np
import shutil

# --- Step 1: Add the 'src' directory to Python's path ---
# This allows us to import the xsnow package from its source directory.
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import xsnow

# --- Step 2: Create a dummy .pro file for the example ---
# This simulates real-world data you would typically have.
pro_data = """
[STATION_PARAMETERS]
StationName= DUMMY_STATION
Latitude= 46.8
Longitude= 9.8
Altitude= 1600
SlopeAngle= 10.0
SlopeAzi= 180.0
[DATA]
0500,15.08.2025 12:00:00
0501,5,0.05,0.15,0.25,0.35,0.45
0502,5,120,150,250,300,320
0500,15.08.2025 13:00:00
0501,6,0.05,0.15,0.25,0.35,0.45,0.55
0502,6,110,140,240,290,310,330
"""
# Place the data in a temporary directory within the examples folder
data_dir = Path("./xsnow_data")
data_dir.mkdir(exist_ok=True)
pro_file = data_dir / "dummy_station.pro"
pro_file.write_text(pro_data)

# --- Step 3: Import the file containing your decorated function ---
# This is the critical step. This line executes the code in example_plugin_analysis.py,
# which runs the @xsnow.register_analysis decorator and populates the plugin registry.
import example_plugin_analysis

# --- Step 4: Load the dataset ---
print("Loading dataset...")
ds = xsnow.xsnowDataset(pro_file)

# --- Step 5: Call the new plugin method ---
# Now that the function has been registered, this call will succeed.
print("\nCalculating mean density of the top 20cm slab...")
top_slab_density = ds.calculate_top_slab_density(depth_cm=20)

# --- Step 6: Print the results ---
if top_slab_density is not None:
    print("\n--- Results ---")
    print(top_slab_density)
    print("\nThis DataArray shows the mean density (in kg/m^3) for each profile")
    print("where the calculation was possible.")

# --- Cleanup the dummy file ---
shutil.rmtree(data_dir)