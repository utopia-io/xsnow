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

In the proposed data model, every dimension is strictly one-dimensional (1D). However, each dimension can be associated with one or more coordinate variables—additional 1D arrays that describe metadata or spatial context (e.g., easting, northing, altitude for the location dimension). This approach keeps the number of dimensions fixed while supporting a wide range of different application cases.

The dimensions of the data model are:

1. Geographic `location`:
    * Specifies the location of the snow profile, further characterized by the coordinates `locationID`, `latitude`, `longitude`, `easting`, `northing`, `altitude` (which are all optional).
    * This setup ensures flexibility to accommodate point locations (e.g., irregular station networks), semi-distributed configurations of topographical classes, and fully-distributed domains (e.g., triangular/hexagonal terrain-adaptive meshes, and regular grids).
    * While this representation is highly flexible, it is not optimal for spatial analyses or map-based visualizations out-of-the-box. To enable 2D spatial plotting or gridded analysis (e.g., of snow surface variables like depth or liquid water content), scalar data variables will first need to be reshaped into 2D grids. See Section on Data manipulation and basic postprocessing for further information.
2. `time`:
    * Represents the specific point in time for which the snow profile or observation is valid.
    * A NumPy array (datetime64[ns]) allows for convenient handling of datetimes with flexible resolutions (e.g., hourly, daily, etc).
3. `slope`:
    * This dimension can (optionally) stratify between different slope and azimuth angles at the same location and time by leveraging the coordinate variables 
        - `inclination` (from horizontal)
        - `azimuth` (orientation from north; e.g., 180° = south).
    * When there is only one angle and azimuth, this case becomes trivial (lengths 1). With several slopes present in the dataset, convenient filtering will be possible (e.g., `ds.sel(slope=(ds.inclination == 30) & (ds.azimuth == 180))`)
4. `realization` instances:
    * This dimension can represent different realizations of the snow profile at the same location and time, such as outputs from:
        - Different weather inputs (e.g., ensemble weather forecasts).
        - Different model configurations (e.g., variations in physics schemes).
        - Observed versus simulated profiles
        - Different subgrid surface cover types (e.g., forest, open glades, etc.).
    * Using this dimension is optional, and the trivial case has only one realization.
    * The coordinate variable `realization` will be stored as integer array, and the description of each integer will be stored as attribute (by the read functions, or manually post-reading).
        - For example: `ds["realization"].attrs["mapping"] = {1: "manual", 2: "simulated"}`
        - A custom labels accessor will allow the user to unambiguously translate realization integers to human-readable labels (e.g., `ds["realization"].labels.get()`).
4. `layer` (i.e, the snow stratigraphy):
    * For every location, time, slope, and realization, there is snow stratigraphy data. Each layer contains attributes specific to a single layer, such as:
         - Height/depth information expressed as `z` (zero at snow surface, negative downward, positive upward)
         - Density, grain size, temperature, ...
    * Any layer attribute will per default refer to the `z` array, but can be pointed to any other vertical reference (e.g., `temperature` could refer to `z_temperature`).
        - This re-directing of the vertical reference to another variable will have to be stored in the variable attributes (e.g., `ds["temperature"].attrs("vertical_reference") = "z_temperature"`) by the read function.
        - A custom `z` accessor will ensure that the user can always unambiguously extract the correct vertical reference for their variable of interest (e.g., `ds["temperature"].z.get()` will return the corresponding values from `z_temperature`).
    * Categorical data variables, like grain types, will be encoded numerically for efficiency reasons. Analogously to mentioned before, the code descriptions will be stored as attribute and can conveniently be queried by the user through the labels accessor (e.g., `ds["grain_code"].labels.get()` would return "PP", "DF", etc.) 
5. `scalar` information:
    * Stores profile-level (not layer-specific) scalar data. Scalars are aggregated values or summary metrics for the profile as a whole. For example, total snow height, etc. This dimension can include forcing data and any scalar output.

To sum up, at a specific *location* and *time*, you may have one or more *realizations*. Each realization represents a complete or partial snow profile, which includes *layer* information (e.g., height, density, etc.) and *scalar* information (forcing and scalar ouput, such as total snow height, etc.). The data structure will be implemented as `xarray.Dataset`.


#### Dataset sparsity
Snow profiles naturally vary in the number of layers they contain, and this number changes over time (e.g., accumulating in winter, decreasing during melt). To maintain a regular gridded structure, all profiles will be padded to a common maximum number of layers using NaN values where no data exists. For some variables (e.g., layer thickness), padding may use 0 instead of NaN to simplify specific analyses.

For example, a dataset with 300 profiles may contain up to 220 layers per profile, but be stored in a fixed array of shape (location, time, slope, realization, layer) with up to 250 layers, and missing entries padded accordingly.

xarray and NumPy handle NaN padding efficiently in vectorized operations, so the impact on performance is minimal—particularly compared to the overhead of managing irregular objects or nested structures. While the resulting dataset may appear sparse, this regularity allows for fast Input/Output, compatibility with NetCDF, simplified filtering, and highly efficient batch processing. The small trade-off in memory overhead is well worth the gain in usability and interoperability.

#### Profile status tracking
To distinguish between locations or time steps with no snow and those with no available data, the dataset will include an integer variable `profile_status`, aligned with the (location, time, slope, realization) dimensions. This variable uses an attribute-based mapping to indicate interpretation: `{
    0: "no data",
    1: "valid profile, no snow",
    2: "valid profile with snow"
}`


#### Class implementation
The data model will be implemented as a custom class that wraps `xarray.Dataset` via composition. A private initialization method will be used internally by all file parsers to standardize and annotate the dataset upon creation. This ensures key validations and metadata assignments are applied consistently, while preserving compatibility with lazy loading. More advanced validation routines can be exposed through a public method if needed. This design is modular, extensible, and future-proof—allowing new methods and features to be added without interfering with the underlying data structure.

#### Metadata: Staying future-proof
International efforts are on the way to establish new standards for a geospatial data format, GeoZarr (see [Open Geospatial Consortium](https://www.ogc.org/announcement/ogc-forms-new-geozarr-standards-working-group-to-establish-a-zarr-encoding-for-geospatial-data/)), which may replace NetCDF in the future. By building on xarray, we are already working towards compatibility. For full compliance, the data model needs to be annotated with CF-style metadata, in particular spatial metadata (i.e., a `crs` attribute) and at least metadata for each coordinate variable.

During implementation, a closer look into the Inishell metadata annotations will provide further guidance as to what metadata to encourage in this project.

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
