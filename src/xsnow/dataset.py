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

1.  **Composition over Inheritance**: This class *contains* an `xarray.Dataset`
    instance (in `self.data`) rather than inheriting from it. This design
    choice, known as composition, is more robust and flexible. It prevents
    future updates to `xarray`'s complex internal machinery from breaking
    this class and gives us full control over the user-facing API.

2.  **Flexible Initialization**: The class can be instantiated directly from
    various sources, including a single file path, a list of files, a
    directory, or even a pre-existing `xarray.Dataset`. This versatility
    is crucial for the package's extension architecture, allowing external
    functions to process data and re-wrap it into a new `xsnowDataset`
    instance.

3.  **Seamless `xarray` Integration**: By using the `__getattr__` magic method,
    the class forwards any unrecognized attribute or method call to the
    underlying `xarray.Dataset`. This allows users to interact with an
    `xsnowDataset` object using the familiar and powerful `xarray` API
    (e.g., `.sel()`, `.mean()`, `.plot`, `.groupby()`).

4.  **Automatic Re-wrapping**: When a forwarded method returns a new
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
import numpy as np

# Use an absolute import assuming the package is installed (e.g., in editable mode)
from xsnow.xsnow_io import read
from xsnow.registry import registry


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

    def __init__(
        self,
        source: Union[str, List[str], xr.Dataset],
        datetime_start: Optional[str] = None,
        datetime_end: Optional[str] = None
    ):
        """
        Initializes the xsnowDataset from a file path, list of paths, directory,
        or an existing xarray.Dataset.

        This flexible constructor is the key hook for extensions. Post-processing
        functions can pass a modified xarray.Dataset to create a new, updated
        xsnowDataset instance.

        Args:
            source (Union[str, List[str], xr.Dataset]): The path to a data file
                (e.g., .pro, .smet, .nc), a directory containing data files, a
                list of file paths, or an existing xarray.Dataset object.
            datetime_start (Optional[str]): If provided, the parser will only include
                profiles with timestamps on or after this date/time.
            datetime_end (Optional[str]): If provided, the parser will only include
                profiles with timestamps on or before this date/time.
        """
        if isinstance(source, xr.Dataset):
            self.data = source
        elif isinstance(source, (str, list)):
            # Use the new high-level read function for parallel I/O, passing filters
            self.data: Optional[xr.Dataset] = read(
                source,
                datetime_start=datetime_start,
                datetime_end=datetime_end
            )
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

    def update(self, source: Union[str, List[str]]):
        """
        Updates the dataset with new data from a source.

        This method is designed for workflows where you load a cached dataset
        (e.g., from NetCDF) and want to append new data from recent .pro or
        .smet files. It finds the last timestamp in the current data and only
        reads new data from the source that is more recent.

        Args:
            source (Union[str, List[str]]): Path to a file or directory, or a
                list of files containing the new data.

        Returns:
            xsnowDataset: A new xsnowDataset instance with the combined data.
        """
        if self.data is None or self.data.time.size == 0:
            # If the current object is empty, this is equivalent to just reading
            return self.__class__(source)

        # Find the last timestamp in the existing data to use as the start filter
        last_timestamp = self.data.time.max().values
        last_timestamp_str = pd.to_datetime(last_timestamp).isoformat()

        print(f"Updating data from {last_timestamp_str}...")
        
        # Read only the new data from the source
        new_data_obj = self.__class__(source, datetime_start=last_timestamp_str)

        if new_data_obj.data is None or new_data_obj.data.time.size == 0:
            print("No new data found to update.")
            return self

        # Combine the old and new datasets
        combined = xr.concat([self.data, new_data_obj.data], dim="time")
        
        # Drop duplicate timestamps, keeping the newest (last) entry
        _, index = np.unique(combined['time'], return_index=True)
        combined = combined.isel(time=index)

        print("Update complete.")
        return self.__class__(combined)

    def merge_smet(self, smet_source: Union[str, List[str]]):
        """
        Reads one or more SMET files and merges their data into this dataset.

        This method is useful for adding meteorological timeseries data to an
        existing dataset of snow profiles. The merge is performed by aligning
        the 'location' and 'time' coordinates.

        Args:
            smet_source (Union[str, List[str]]): Path to a SMET file, a list of
                SMET files, or a directory containing SMET files.

        Returns:
            xsnowDataset: A new xsnowDataset instance with the merged data.
        """
        if self.data is None:
            print("Cannot merge into an empty dataset. Load data first.")
            return self

        smet_data = read(smet_source)
        if smet_data is None:
            print("No valid SMET data found to merge.")
            return self

        # Merge the datasets. xarray will automatically align on common coordinates.
        merged_ds = xr.merge([self.data, smet_data])
        return self.__class__(merged_ds)

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

    def to_pro(self, output_dir: str, max_profiles: int = 100, **kwargs):
        """
        Saves the dataset to one or more .pro files.

        This method iterates through each profile in the dataset (each unique
        combination of location and time) and writes a separate .pro file
        for each one in the specified output directory.

        To prevent accidentally writing a huge number of files, this function
        will raise a ValueError if the total number of profiles exceeds
        `max_profiles`.

        Args:
            output_dir (str): The path to the directory where .pro files will be saved.
            max_profiles (int): The maximum number of .pro files to generate.
                Defaults to 100.
            **kwargs: Placeholder for future filtering options.
        """
        if self.data is None:
            print("No data to save.")
            return

        num_locations = self.data.sizes.get('location', 1)
        num_times = self.data.sizes.get('time', 1)
        total_profiles = num_locations * num_times

        if total_profiles > max_profiles:
            raise ValueError(
                f"The dataset contains {total_profiles} profiles, which exceeds the "
                f"limit of {max_profiles}. To proceed, please specify a higher "
                f"limit using the `max_profiles` argument."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a reverse mapping from standard name to parameter code
        name_to_code_map = {
            name: details['param_code']
            for name, details in registry.items() if 'param_code' in details
        }

        print(f"Exporting {total_profiles} profiles to {output_dir}...")
        # Iterate through each location and time to write individual profiles
        for loc in self.data.location.values:
            for t in self.data.time.values:
                profile = self.data.sel(location=loc, time=t)
                
                # Format timestamp for filename and data section
                timestamp_obj = pd.to_datetime(t)
                ts_filename = timestamp_obj.strftime('%Y%m%d_%H%M%S')
                ts_data = timestamp_obj.strftime('%d.%m.%Y %H:%M:%S')
                
                filename = f"{loc}_{ts_filename}.pro"
                filepath = output_path / filename

                with open(filepath, 'w') as f:
                    # Write Header
                    f.write("[STATION_PARAMETERS]\n")
                    f.write(f"StationName= {loc}\n")
                    f.write(f"Latitude= {profile.latitude.item()}\n")
                    f.write(f"Longitude= {profile.longitude.item()}\n")
                    f.write(f"Altitude= {profile.altitude.item()}\n")
                    if 'slope' in profile.coords:
                        f.write(f"SlopeAngle= {profile.slope.item()}\n")
                    if 'slope_aspect' in profile.coords:
                        f.write(f"SlopeAzi= {profile.slope_aspect.item()}\n")
                    
                    # Write Data
                    f.write("[DATA]\n")
                    f.write(f"0500,{ts_data}\n")

                    # Iterate through data variables to write layer data
                    for var_name in sorted(profile.data_vars):
                        if var_name in name_to_code_map:
                            param_code = name_to_code_map[var_name]
                            # Get the layer data, drop NaNs, and format
                            layer_data = profile[var_name].dropna(dim='layer').values
                            if layer_data.size > 0:
                                num_layers = len(layer_data)
                                data_str = ",".join(map(str, layer_data))
                                f.write(f"{param_code},{num_layers},{data_str}\n")
        print("Export complete.")

    def to_smet(self, output_dir: str, max_files: int = 100, **kwargs):
        """
        Saves the timeseries data for each location to a .smet file.

        This method extracts variables that only vary with time (not by layer)
        for each location and writes them to a separate .smet file.

        To prevent accidentally writing a huge number of files, this function
        will raise a ValueError if the number of locations exceeds `max_files`.

        Args:
            output_dir (str): The path to the directory where .smet files will be saved.
            max_files (int): The maximum number of .smet files to generate.
                Defaults to 100.
            **kwargs: Placeholder for future filtering options.
        """
        if self.data is None:
            print("No data to save.")
            return

        num_locations = self.data.sizes.get('location', 1)
        if num_locations > max_files:
            raise ValueError(
                f"The dataset contains {num_locations} locations, which exceeds the "
                f"limit of {max_files}. To proceed, please specify a higher "
                f"limit using the `max_files` argument."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting {num_locations} SMET files to {output_dir}...")
        for loc in self.data.location.values:
            location_ds = self.data.sel(location=loc)
            
            # Filter for timeseries variables (those without a 'layer' dimension)
            timeseries_vars = [var for var, da in location_ds.data_vars.items() if 'layer' not in da.dims]
            if not timeseries_vars:
                print(f"Skipping location '{loc}': No timeseries data found.")
                continue

            df = location_ds[timeseries_vars].to_dataframe()
            
            # SMET format requires a 'timestamp' column
            df = df.reset_index().rename(columns={'time': 'timestamp'})
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

            filename = f"{loc}.smet"
            filepath = output_path / filename

            with open(filepath, 'w') as f:
                # Write Header
                f.write("SMET 1.1 ASCII\n")
                f.write("[HEADER]\n")
                f.write(f"station_name     = {loc}\n")
                f.write(f"latitude         = {location_ds.latitude.item()}\n")
                f.write(f"longitude        = {location_ds.longitude.item()}\n")
                f.write(f"altitude         = {location_ds.altitude.item()}\n")
                f.write(f"nodata           = -999\n")
                f.write(f"fields           = {' '.join(df.columns)}\n")
                f.write("[DATA]\n")
                
                # Write data
                df.to_csv(f, sep=' ', index=False, header=False, lineterminator='\n')
        print("Export complete.")


    def validate_units(self):
        """
        Checks if the units of variables in the dataset match the SI units
        defined in the central registry.
        
        Prints a warning for any variable with non-compliant units.
        """
        if self.data is None:
            print("Dataset is not loaded. Cannot validate units.")
            return
            
        print("--- Running Unit Validation ---")
        for var_name, var_data in self.data.data_vars.items():
            if var_name in registry:
                expected_units = registry[var_name].get("si_units")
                actual_units = var_data.attrs.get("units")
                
                if actual_units != expected_units:
                    print(
                        f"Warning: Unit mismatch for '{var_name}'. "
                        f"Expected: '{expected_units}', Found: '{actual_units}'"
                    )
        print("--- Validation Complete ---")

    def compute_hs(self):
        """
        Computes the total snow height (HS) from layer heights.
        
        This method calculates the maximum height for each profile and adds it
        to the dataset as a new variable 'HS'.
        
        Returns:
            xsnowDataset: A new xsnowDataset instance with the 'HS' variable.
        """
        if self.data is None:
            raise ValueError("Cannot compute HS on an empty dataset.")
        if 'height' not in self.data:
            raise ValueError("Cannot compute HS without 'height' variable.")
            
        hs = self.data['height'].max(dim='layer', skipna=True)
        hs.attrs = {
            'long_name': 'Total Snow Height',
            'units': 'm'
        }
        
        new_ds = self.data.assign({'HS': hs})
        return self.__class__(new_ds)

    def compute_hn24(self):
        """
        Computes the 24-hour change in snow height (HN24).
        
        This method calculates the difference in total snow height over a
        24-hour period for each location. It requires the 'HS' variable
        to be present, which can be calculated with `.compute_hs()`.
        
        Returns:
            xsnowDataset: A new xsnowDataset instance with the 'HN24' variable.
        """
        if self.data is None:
            raise ValueError("Cannot compute HN24 on an empty dataset.")
        if 'HS' not in self.data:
            raise ValueError("Missing 'HS' variable. Run .compute_hs() first.")
        
        # Resample to 1-hour frequency to handle potential gaps, then calculate diff
        # Use 'h' for hourly frequency to avoid FutureWarning
        hn24 = self.data['HS'].resample(time='1h').interpolate('linear').diff(dim='time', n=24)
        hn24 = hn24.reindex_like(self.data, method='nearest')
        
        hn24.attrs = {
            'long_name': '24-hour Change in Snow Height',
            'units': 'm'
        }
        
        new_ds = self.data.assign({'HN24': hn24})
        return self.__class__(new_ds)

    def to_gpu(self):
        """
        Moves the dataset's data to the GPU using CuPy if available.

        This method checks for an available GPU and the `cupy` library. If found,
        it converts the underlying NumPy arrays to CuPy arrays for accelerated
        computation.

        Returns:
            xsnowDataset: A new xsnowDataset instance with data on the GPU, or
            the original object if a GPU is not available.
        """
        if self.data is None:
            print("No data to move to GPU.")
            return self

        try:
            import cupy as cp
            cp.arange(1) # Check if a GPU is present and cupy is working
        except (ImportError, RuntimeError):
            print("INFO: Cupy not found or GPU not available. Returning original dataset.")
            return self

        print("Moving data to GPU...")
        # Create a new dataset by converting each DataArray's data to a cupy array
        gpu_ds = self.data.copy(deep=False)
        # FIX: Iterate over data_vars only, not all variables (which includes coordinates)
        for var in gpu_ds.data_vars:
            # Only convert numeric arrays, skip others (like datetime)
            if np.issubdtype(gpu_ds[var].dtype, np.number):
                 gpu_ds[var].data = cp.asarray(gpu_ds[var].data)

        return self.__class__(gpu_ds)

    def to_cpu(self):
        """
        Moves the dataset's data back to the CPU (NumPy).

        This is necessary before performing operations that are not supported
        on the GPU, such as saving to NetCDF.

        Returns:
            xsnowDataset: A new xsnowDataset instance with data on the CPU.
        """
        if self.data is None:
            print("No data to move to CPU.")
            return self

        # xarray's as_numpy() is hardware-aware and will convert from cupy to numpy
        print("Moving data to CPU...")
        cpu_data = self.data.as_numpy()
        return self.__class__(cpu_data)
