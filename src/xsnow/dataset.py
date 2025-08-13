# dataset.py
"""
Defines the core xsnowDataset class for the xsnow package.

This module contains the primary user-facing object, `xsnowDataset`. It is
designed as a high-level, convenient wrapper around a standard `xarray.Dataset`
object, tailored for handling multi-dimensional snow profile data.

Core Design Philosophy
----------------------
The `xsnowDataset` class aims to simplify the user experience without
sacrificing the power and flexibility of the underlying xarray data model.
It achieves this through several key features:

1.  **Flexible Initialization**: The class can be instantiated directly from
    various sources, including a single file path, a list of files, a
    directory, or even a pre-existing `xarray.Dataset`. This versatility
    is crucial for the package's extension architecture, allowing external
    functions to process data and re-wrap it into a new `xsnowDataset`
    instance.

2.  **Seamless `xarray` Integration**: By using the `__getattr__` magic method,
    the class forwards any unrecognized attribute or method call to the
    underlying `xarray.Dataset`. This allows users to interact with an
    `xsnowDataset` object using the familiar and powerful `xarray` API
    (e.g., `.sel()`, `.mean()`, `.plot`, `.groupby()`).

3.  **Automatic Re-wrapping**: When a forwarded method returns a new
    `xarray.Dataset`, the class automatically wraps the result back into an
    `xsnowDataset` instance. This ensures that the user can chain operations
    (e.g., `ds.sel(...).mean(...)`) and always retain the convenience of the
    `xsnowDataset` wrapper.

This design makes the `xsnowDataset` the central component for both interactive
analysis and the development of advanced post-processing extensions.
"""

from typing import Optional, Union, Any, List
from pathlib import Path

import pandas as pd
import xarray as xr

# Use an absolute import assuming the package is installed (e.g., in editable mode)
from xsnow.xsnow_io import read


class xsnowDataset:
    """
    A class to represent and interact with a multi-dimensional snow profile dataset.

    This class acts as a high-level wrapper around an xarray.Dataset, providing
    a convenient way to load data from various file formats. It forwards attribute
    and method calls to the underlying xarray.Dataset, re-wrapping the results
    when appropriate. This design enables extension via external functions or
    decorator classes.

    Attributes:
        data (Optional[xr.Dataset]): The underlying xarray.Dataset that holds
            all the snow profile data and metadata. It is None if loading fails.
    """

    def __init__(self, source: Union[str, List[str], xr.Dataset]):
        """
        Initializes the xsnowDataset from a file path, list of paths, directory,
        or an existing xarray.Dataset.

        This flexible constructor is the key hook for extensions. Post-processing
        functions can pass a modified xarray.Dataset to create a new, updated
        xsnowDataset instance.

        Args:
            source (Union[str, List[str], xr.Dataset]): The path to a data file,
                a directory containing .pro files, a list of file paths, or an
                existing xarray.Dataset object.
        """
        if isinstance(source, xr.Dataset):
            self.data = source
        elif isinstance(source, (str, list)):
            # Use the new high-level read function for parallel I/O
            self.data: Optional[xr.Dataset] = read(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def __repr__(self) -> str:
        """
        Provides a developer-friendly, summary representation of the dataset.
        """
        if self.data is None:
            return "<xsnowDataset (empty)>"

        # Dynamically create a summary string
        try:
            locations = self.data.sizes.get('location', 0)
            times = self.data.sizes.get('time', 0)
            start_time = pd.to_datetime(self.data.time.min().values).strftime('%Y-%m-%d') if times > 0 else 'N/A'
            end_time = pd.to_datetime(self.data.time.max().values).strftime('%Y-%m-%d') if times > 0 else 'N/A'
            
            return (
                f"<xsnowDataset>\n"
                f"  Locations: {locations}\n"
                f"  Profiles: {times}\n"
                f"  Time Range: {start_time} to {end_time}"
            )
        except Exception:
            return f"<xsnowDataset with data>"

    def __getattr__(self, name: str) -> Any:
        """
        Forwards attribute access to the underlying xarray.Dataset.

        This method is the hook that allows extensions and users to interact
        with the object as if it were a native xarray.Dataset (e.g., calling
        .sel(), .mean(), accessing .sizes). If a method call returns a new
        xarray.Dataset, it is automatically re-wrapped in the xsnowDataset class.
        """
        if self.data is None:
            raise AttributeError("Dataset is not loaded.")
        
        attr = getattr(self.data, name)

        if callable(attr):
            # This is a method (like .sel(), .mean(), etc.)
            def method_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If the result is a new Dataset, re-wrap it in our class
                if isinstance(result, xr.Dataset):
                    return self.__class__(result)
                # Otherwise, return the result as-is (e.g., a numpy array, a float)
                return result
            return method_wrapper
        else:
            # This is a property (like .sizes, .coords, etc.)
            return attr

    def to_netcdf(self, output_path: str):
        """
        Saves the dataset to a NetCDF file.

        This method provides a convenient way to cache processed data or export
        it for use in other applications that support NetCDF.

        Args:
            output_path (str): The destination file path for the .nc file.
        """
        if self.data is None:
            print("No data to save.")
            return

        try:
            # Ensure the parent directory exists before trying to save the file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Use xarray's built-in method to save the data
            self.data.to_netcdf(output_path)
            print(f"Dataset successfully saved to {output_path}")
        except Exception as e:
            print(f"Failed to save NetCDF file: {e}")


# Example Usage (for testing purposes):
if __name__ == '__main__':
    # NOTE: To run this example block directly, you must have the `xsnow`
    # package installed in your environment, for instance, by running
    # `uv pip install -e .` from the project's root directory.

    # 1. Create dummy .pro files to test with
    dummy_files = []
    for i in range(3):
        dummy_pro_content = f"""
[STATION_PARAMETERS]
StationName= DUMMY_STATION_{i}
Latitude= {46.8 + i*0.1}
Longitude= {9.8 + i*0.1}
Altitude= {2540 + i*10}
SlopeAngle= 30.0
SlopeAzi= 180.0
[DATA]
0500,13.08.2025 12:00:00
0501,2,0.1,0.2
0502,2,250,300
"""
        dummy_filepath = f"dummy_station_{i}.pro"
        with open(dummy_filepath, "w") as f:
            f.write(dummy_pro_content)
        dummy_files.append(dummy_filepath)

    print(f"--- Initializing xsnowDataset with a list of files ---")
    
    # 2. Instantiate the main class with the list of files
    snow_data = xsnowDataset(dummy_files)

    # 3. Print the object's representation
    print("\n--- Object Representation ---")
    print(snow_data)

    # 4. Access the underlying xarray.Dataset
    if snow_data.data:
        print("\n--- Accessing underlying xarray.Dataset ---")
        print("Dataset sizes:", snow_data.sizes)
        print("Location coordinates:", snow_data.coords['location'].values)
        

    # 5. Clean up the dummy files
    for f in dummy_files:
        Path(f).unlink()
