# An Open-Source Collaborative Python Package for Parsing Snow Cover Simulations
---
## Background
One-dimensional snow cover models, such as SNOWPACK and CROCUS, simulate the stratigraphy of the seasonal snow cover. These models offer an opportunity for avalanche warning services to complement traditional, often sparse, information sources and provide an independent second opinion. Recent years have seen exciting developments in this field, with several community-driven initiatives gaining momentum. Notably, the international working group [AvaCollabra](www.avacollabra.org) and the open-source software framework [AWSOME](https://gitlab.com/avalanche-warning) demonstrate concerted efforts to collaboratively advance the development and applications of these models.

Inspired by recent research developments, avalanche warning services around the world have invested in the new technology and are drafting ambitious visions for their operational modeling suites, working towards semi-automation and gridded products. Collaboration on common processing pipelines has started with AWSOME, which offers solutions for preprocessing weather inputs, running snowpack simulations, postprocessing the raw simulations, and visualizing the results. However, there is currently no widely adopted software package that standardizes the link between raw snow cover simulation outputs and postprocessing algorithms. Currently, every stakeholder has developed their own set of parsers, almost exclusively using inefficient text-based data files and non-vectorized data models (with the exception of Meteo France). This fundamentally limits the operational, large-scale application of these simulations, particularly in large forecasting regions or for moving towards high-resolution or ensemble simulations. In addition, sharing new postprocessing algorithms among different stakeholders is currently quite challenging, which makes it difficult and slow to turn research developments into operational applications.


### Objective
The objective of this project is to develop an open-source Python package that efficiently parses and manipulates large snow cover simulation datasets. This package will bridge the gap between raw simulation data and advanced postprocessing workflows by:
- Importing and exporting simulation data in efficient, standardized formats.
- Providing basic data manipulation methods.
- Implementing an internal gridded data model powered by high-performance libraries like `xarray` and `numpy`.
- Ensuring seamless integration with efficient data storage formats, such as databases or NetCDF files.

The general scope of the package focuses on simulated snow profiles, with the option to handle observed profiles to facilitate comparisons and the initialization of simulations with observed profiles. The package design caters to both operational and research needs, maintaining a balance between an intuitive implementation to maximize accessibility and efficient processing to support mid-sized operational agencies and mature research projects.

### Novelty
This project addresses a major development gap that hinders not only the scalability of operational modeling efforts but also the exchange of postprocessing algorithms among different researchers and avalanche warning agencies. Through collaborative development, this project ensures that the needs of all types of stakeholders are met which is critical for wide acceptance of the package. This software implementation will establish the technical groundwork for efficiently analyzing snow cover simulations across large domains. Moreover, it will provide a clear interface for SNOWPACK maintainers to develop output plugins, such as NetCDF writers, to directly support gridded data structures. This forward-looking capability will further enhance efficiency in future applications.

---
## Internal data model
Unlike data models that treat each snow profile as an independent object, the proposed model is built on a gridded data structure organized along several dimensions. This structure leverages the high-performance library `xarray` to enable efficient vectorized operations, memory management (e.g., lazy loading and batch processing), and scalability. Metadata---including units, variable descriptions, time zones, and coordinate reference systems---can be embedded to ensure data consistency and ease of use.

The dimensions of the data model are:

1. Geographic location:
    * Specifies the location of the snow profile, defined by 1D to 3D coordinates (e.g., station ID, x/y/z, latitude/longitude/altitude).
    * Notes: Ensure flexibility to accommodate point locations (e.g., a weather station) and gridded domains (e.g., a spatially distributed model).
2. Time:
    * Represents the specific point in time for which the snow profile or observation is valid.
    * Notes: Allow for datetime objects with flexible time resolutions (e.g., hourly, daily, etc).
3. Virtual slopes (optional dimension):
    * Represents different slope angles/azimuth at the same location and time
    * This dimension is optional and is trivial when there is only one combination of angle/azimuth.
    * Is one dimension `vslopes` enough to catch different combinations of angles/azimuth, or do we want two separate dimensions?
4. Realization instances (optional dimension):
    * Represents different realizations of the snow profile at the same location and time, such as outputs from:
        - Different weather inputs (e.g., ensemble weather forecasts).
        - Different model configurations (e.g., variations in physics schemes).
        - Observed versus simulated profiles.
    * Notes: This dimension is optional and should only be included if multiple realizations exist.
4. Profile structure (i.e, snow layers):
    * A hierarchical dimension within each realization that represents multiple layers in the snowpack. Each layer contains attributes specific to a single layer, such as:
         - Height (& depth)
         - Density, grain size, ...
         - Temperature (may need its own vertical temperature profile spacing)
    * Notes: This hierarchical structure ensures that the layers are indexed within each profile and realization.
5. Scalar information:
    * Stores profile-level (not layer-specific) scalar data, such as total snow height, etc. It also includes the weather forcing.
    * Notes: Scalars are aggregated values or summary metrics for the profile as a whole.

To sum up, at a specific *location* and *time*, you may have one or more *realizations* (optional, depending on the scenario). Each realization represents a complete or partial snow profile, which includes *layer* information (e.g., height, density, etc.) and *scalar* information (e.g., total snow height, etc.). The data structure will be implemented as `xarray.Dataset`.


###### Comment on dataset sparsity
Snow profiles inherently contain different numbers of layers, and layer numbers will vary throughout the season. The dataset will therefore include NaN values for missing layers in profiles with fewer layers. xarray handles NaN values gracefully in vectorized operations to keep impact on processing efficiency minimal. Therefore, while the resulting dataset may appear sparse, the benefits of regular gridding---such as fast read times and streamlined analysis---far outweigh the drawbacks.

---
## Input/Output parsers
The package will provide read and write routines for various file formats. Hence, the internal data model can be populated by and saved as:

* SNOWPACK: Native PRO and SMET formats.
* CROCUS: NetCDF format.
* CAAML and JSON: Formats used for manual snow profile observations or snow penetrometer measurements.
* Custom NetCDF: Optimized for the package's internal data structure.

### API
The API will follow conventions used by `pandas` and `xarray` by designing *read function* and *write methods*. `read_*()` functions will return a simulation dataset (i.e., `xarray.Dataset`), whereas write methods `.to_*()` applicable to the dataset will allow writing to disk.

On an abstract level, file conversion could therefore look like this:

```python
# Read a single SNOWPACK PRO file
ds = read_pro("example.pro")

# Convert it to another format
ds.to_netcdf("output.nc")
ds.to_caaml("output.caaml")
```

#### Read function signatures
The main focus will lie on read functions for SNOWPACK and CROCUS formats, while providing a basic functionality to read CAAML and JSON formats. The function signatures could look like this:

```python
import datetime
import xarray
from typing import Optional, List
```
```python
def read_pro(
    filepath: str,
    timestamps: Optional[List[datetime.datetime]] = None,
    time_resolution: Optional[int] = None,
    remove_soil: bool = True,
    consider_surfacehoar: bool = True
) -> xarray.Dataset:
    """
    Reads a SNOWPACK PRO file into an xarray Dataset.

    Parameters:
    -----------
        filepath (str):
            Path to the PRO file.

        timestamps (Optional[List[datetime.datetime]]):
            An optional list of datetime objects to filter the dataset to specific timestamps.

        time_resolution (Optional[int]):
            If timestamps is None, you can specify the time resolution in hours.

        remove_soil (bool):
            If True, removes soil layers from the profiles.

        consider_surfacehoar (bool):
            If True, surface hoar layers modeled at the surface will explicitly
            be added as topmost layer of the respective profile. Several layer
            properties of this layer will then default to None.
    """
    pass
```
```python
def read_smet(
    filepath: str,
    timestamps: Optional[List[datetime.datetime]] = None
) -> xarray.Dataset:
    """
    Reads a SNOWPACK SMET file into an xarray Dataset.

    Parameters:
    -----------
        filepath (str):
            Path to the SMET file.

        timestamps (Optional[List[datetime.datetime]]):
            An optional list of datetime objects to filter the dataset to specific timestamps.
    """
    pass
```
```python
# @Leo Viallon: Your input required
def read_crocus(filepath: str) -> xarray.Dataset:
    pass

def read_json(filepath: str) -> xarray.Dataset:
    pass
```
```python
# @Michi Binder, others: input required
def read_caaml(filepath: str) -> xarray.Dataset:
    pass
```

There will also be a wrapper that can read multiple files and formats into a single dataset.

```python
def read(directories: Optional[List[str]]
         filepaths: Optional[List[str]],
         **kwargs
) -> xarray.Dataset:
    """
    Reads multiple snow cover simulations into an xarray Dataset.

    Iterates through all files within `directories` and `filenames`,
    auto-detects which read routine to call, and assembles one large
    simulation dataset. If there are multiple profiles at the same location
    and time, the different profiles will be stored as separate realizations.

    Parameters:
    -----------
        directories (Optional[List[str]]):
            Paths to directories, of which every file should be read.

        filepaths (Optional[List[str]]):
            Paths to individual files that will be read.

        **kwargs:
            Keyword arguments passed to the individual read functions.

    Examples:
    ---------
        >>> # Read SNOWPACK simulations generated with two different configurations
        >>> #     into one simulation dataset with two realizations:
        >>> ds = read(['./snowpack_A/', './snowpack_B/'])
    """
    pass
```


###### To be determined

- [ ] Strategy for error handling for corrupted/incomplete datasets
- [ ] Strategy for validation checks (e.g., units, etc.)

#### Write function signatures
For every `read_*()` function will be an equivalent `.to_*()` method.

---
## Data manipulation and basic postprocessing
In addition to xarrays's functionality, several data manipulation wrappers and convenience methods may be required, for example:

* Combining of datasets
  - along one or multiple dimensions (e.g., space, time, realizations)
  - updating of entries (i.e., overwriting of existing data fields)
* Filtering, subsetting, extraction of sets of scalars/layers from multiple profiles


Lastly, to illustrate the power of the gridded data structure, the package will implement a small selection of simple postprocessing routines, such as analytical stability indices and snowpack metrics (e.g., change of 24 hour snow height, slab density, liquid water content index, etc.).

*To be determined in more detail at a later stage.*
