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
    * When there is only one angle and azimuth, this case becomes trivial (lengths 1). With several slopes present in the dataset, convenient subsetting will be possible (e.g., `ds.sel(slope=(ds.inclination == 30) & (ds.azimuth == 180))`)
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
5. Scalar information: (not its own dimension, but mentioned for comprehensiveness)
    * Stores profile-level (not layer-specific) scalar data. Scalars are aggregated values or summary metrics for the profile as a whole. For example, total snow height, etc. Scalar data variables can include forcing data and any scalar output.

To conclude, at a specific *location* and *time*, you may have one or more *realizations* for each *slope* configuration of inclination and azimuth. Each realization represents a complete or partial snow profile. The snow profile data is stored in data variables. Some data variables represent scalar information for the profile as a whole. Other data variables characterize specific *layer* properties (e.g., height, density, ...). 

 * Dimensions of each scalar data variable: `(location, time, slope, realization)` 
 * Dimensions of each layer property: `(location, time, slope, realization, layer)`. 



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

The initialization routine needs to initialize a fixed number of layers, which should be set to one of i) a default value, ii) the maximum number of layers in the dataset to be read, iii) a user-provided value. The user needs to be able to strategically decide on how often to grow the dataset, such as in operational contexts where profiles may grow layers each day.

#### Metadata: Staying future-proof
International efforts are on the way to establish new standards for a geospatial data format, GeoZarr (see [Open Geospatial Consortium](https://www.ogc.org/announcement/ogc-forms-new-geozarr-standards-working-group-to-establish-a-zarr-encoding-for-geospatial-data/)), which may replace NetCDF in the future. By building on xarray, we are already working towards compatibility. For full compliance, the data model needs to be annotated with CF-style metadata, in particular spatial metadata (i.e., a `crs` attribute) and at least metadata for each coordinate variable.

During implementation, a closer look into the Inishell metadata annotations will provide further guidance as to what metadata to encourage in this project.


#### Naming convention of data variables
While the names of the dimensions, coordinate variables, and a few other variables (like `z`) have already been decided on, there are many data variables that need to be named. This package will maintain a canonical registry of variable names that will list every variable that our package "knows" along with its *aliases*, *CF standard names*, and *SI units*. Each read routine will rename all aliases of data variables that it finds in a file to their canonical names and attach standard names and units as attributes.

Upon start of the package implementation, this registry will grow naturally until a community decision on canonical names and relevant aliases needs to be made (before first official release).

```python
# Example of name registry.py
registry = {
    "density": {
        "aliases"      : ["density", "rho", "rho_snow"],
        "standard_name": "mass_density_of_snow",
        "si_units"     : "kg m-3"
    }
}
```

---
## Input/Output parsers
The package will provide read and write routines for various file formats. Hence, the internal data model can be populated by and saved as:

* SNOWPACK: Native PRO and SMET formats. 
* CROCUS: NetCDF format.
* CAAML SnowProfile format and its JSON representation: Formats used for manual snow profile observations or snow penetrometer measurements. Implementation through brandnew [Meteo FR package](https://github.com/UMR-CNRM/snowprofile/).
* Custom NetCDF: Optimized for the package's internal data structure.

### API
The API will follow conventions used by `pandas` and `xarray` by designing *read function* and *write methods*. `read_*()` functions will return a simulation dataset (i.e., `xarray.Dataset`), whereas write methods `.to_*()` applicable to the dataset will allow writing to disk.

On an abstract level, file conversion could therefore look like this:

```python
# Read a single SNOWPACK PRO file
ds = read_pro("example.pro")

# Convert it to another format
ds.to_netcdf("output.nc")
```

#### Read function signatures
The main focus will lie on read functions for SNOWPACK and CROCUS formats, while providing a basic functionality to read CAAML and JSON formats. A read and write routine for a custom NetCDF file that resembles this package's data structure will also be included. 

While this package strives to separate basic tasks into different routines (e.g., reading versus subsetting versus validating), the read functions will contain some subsetting capabilities to optimize efficiency. The parsers for text-based SNOWPACK formats will allow for simple time subsetting. The parsers of binary dataset files will allow for subsetting all main dimensions. For convenience, there will be a high-level `read` routine that can automatically choose the appropriate parser and will also allow for subsetting all main dimensions.

The function signatures could look like this:

```python
import datetime
import xarray
from typing import Optional, Union, Sequence
```
##### SNOWPACK formats
```python
def read_pro(
    filepath: str,
    datetime_start: Optional[datetime.datetime] = None,
    remove_soil: bool = True,
    include_smet: bool = True
) -> xarray.Dataset:
    """
    Reads a SNOWPACK PRO file into an xarray Dataset.

    Parameters:
    -----------
        filepath : str
            Path to the PRO file.

        datetime_start : datetime.datetime, optional
            The PRO file will be read starting with this datetime. Data prior to 
            this timestamp will be ignored. This feature optimizes efficiency
            when reading only recentmost timestamps in an operational day-to-day
            setting.

        remove_soil : bool
            If True, removes soil layers from the profiles.

        include_smet : bool
            If True, also try reading the corresponding SMET file by swapping
            file extensions from .pro to .smet.

    """
    pass

```
If a surface hoar (SH) layer is present at the surface, SNOWPACK will not include the layer as the topmost layer, but as scalar meta-information. In `read_pro()`, we discard this peculiarity and enforce SH layers at the surface as standalone topmost layer. Since this SH layer won't contain all properties, missing SH properties will default to None in this case. Implementation example: argument `consider_surfacehoar` in `snowpacktools`'s `read_pro` function.

```python
def read_smet(
    filepath: str
) -> xarray.Dataset:
    """
    Reads a SNOWPACK SMET file into an xarray Dataset.

    Parameters:
    -----------
        filepath (str):
            Path to the SMET file.
    """
    pass
```

##### CROCUS formats
Similarly to SNOWPACK's PRO and SMET files, the forcing and output data of CROCUS simulations may also be stored in two separate files. For more information, consult Leo Viallon-Galinier.
```python
def read_crocus(filepath: Union[str, Sequence[str]],
                *,
                time: Optional[TimeSelector] = None,
                location: Optional[LocationSelector] = None,
                slope: Optional[SlopeSelector] = None,
                realization: Optional[RealizationSelector] = None,
                **open_kwargs) -> xarray.Dataset:
    """
    Read one or more CROCUS NetCDF files into an xarray.Dataset with lazy I/O and optional pre-subsetting.

    Parameters
    ----------
    files : str or sequence of str
        Path or list of paths to CROCUS NetCDF file(s).
    time : TimeSelector, optional
        Temporal selector (e.g., datetime, slice, list of datetimes).
    location : LocationSelector, optional
        Spatial selector (e.g., station ID, slice, list of IDs).
    slope : SlopeSelector, optional
        Slope selector (e.g., index, slice, list of indices, or boolean mask).
    realization : RealizationSelector, optional
        Realization selector (e.g., index, slice, list of indices).
    **open_kwargs
        Additional keyword arguments forwarded to xarray’s open_dataset/open_mfdataset.

    Returns
    -------
    xr.Dataset
        Merged and subsetted dataset.
    """
    pass
```

##### This package's file format
```python
def read_netcdf(filepath: Union[str, Sequence[str]],
                *,
                time: Optional[TimeSelector] = None,
                location: Optional[LocationSelector] = None,
                slope: Optional[SlopeSelector] = None,
                realization: Optional[RealizationSelector] = None,
                **open_kwargs) -> xarray.Dataset:
    """
    Read one or more NetCDF files that represent the internal data structure of this package into an xarray.Dataset with lazy I/O and optional pre-subsetting.

    Parameters
    ----------
    files : str or sequence of str
        Path or list of paths to NetCDF file(s).
    time : TimeSelector, optional
        Temporal selector (e.g., datetime, slice, list of datetimes).
    location : LocationSelector, optional
        Spatial selector (e.g., station ID, slice, list of IDs).
    slope : SlopeSelector, optional
        Slope selector (e.g., index, slice, list of indices, or boolean mask).
    realization : RealizationSelector, optional
        Realization selector (e.g., index, slice, list of indices).
    **open_kwargs
        Additional keyword arguments forwarded to xarray’s open_dataset/open_mfdataset.

    Returns
    -------
    xr.Dataset
        Merged and subsetted dataset.
    """
    pass

    ## Example implementation of binary parsers:
    # 1) collect only the filters the user actually passed
    filter_args = {
        "time": time,
        "location": location,
        "slope": slope,
        "realization": realization,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}

    # 2) build a preprocess function that applies your generic _subset()
    preprocess = partial(_subset, **filter_args) if filter_args else None

    # 3) when given multiple files, use open_mfdataset + preprocess
    if isinstance(files, (list, tuple)):
        ds = xr.open_mfdataset(
            files,
            preprocess=preprocess,
            combine="by_coords",
            **open_kwargs
        )
    else:
        # single file: open, then apply the same subset
        ds = xr.open_dataset(files, **open_kwargs)
        if preprocess:
            ds = preprocess(ds)

    return(ds)
```

##### Manual observations
```python
def read_caaml(filepath: str) -> xarray.Dataset:
    # Implementation through PyPi package snowprofile
    pass

def read_json(filepath: str) -> xarray.Dataset:
    # Implementation through PyPi package snowprofile 
    # at the earlierst when JSON parser exists in there
    pass
```

##### High-level wrapper
There will also be a wrapper that can read multiple files and formats into a single dataset. This wrapper will allow for convenient subsetting of the main dimensions. 

```python
def read(
    filepaths: Optional[Union[str, Sequence[str]]] = None,
    *,
    directories: Optional[Union[str, Sequence[str]]] = None,
    time: Optional[TimeSelector] = None,
    location: Optional[LocationSelector] = None,
    slope: Optional[SlopeSelector] = None,
    realization: Optional[RealizationSelector] = None,
    **parser_kwargs
) -> xr.Dataset:
    """
    Reads multiple snow cover simulation outputs into a single xarray.Dataset.

    This function iterates through all specified `filepaths` and all supported
    files found in each `directories` path, auto-detects the appropriate parser
    for each format, and merges the results into one consolidated dataset.
    The output can be subsetted along time, location, slope configuration, or
    realization. If multiple profiles exist at the same (location, time), they 
    are stored as separate realizations.

    Parameters
    ----------
    filepaths : str or list of str, optional
        Path or list of paths to individual simulation files to read.
    directories : str or list of str, optional
        Path or list of directories whose supported files will all be read.
    time : TimeSelector, optional
        Selector for the time dimension (e.g., datetime, slice, list of
        datetimes, or pandas.DatetimeIndex).
    location : LocationSelector, optional
        Selector for the location dimension (e.g., single ID string, slice,
        or list of strings).
    slope : SlopeSelector, optional
        Selector for the slope dimension (e.g., integer index, slice,
        list of indices, boolean mask or DataArray).
    realization : RealizationSelector, optional
        Selector for the realization dimension (e.g., integer index, slice,
        or list of integers).
    **parser_kwargs
        Additional keyword arguments forwarded to the individual parsers.

    Returns
    -------
    xr.Dataset | SnowProfileDataset
        Consolidated and optionally subsetted dataset containing snow profile
        simulations, with dimensions for location, time, slope, realization,
        layer, and scalar.

    Examples
    --------
    >>> ds = read("profile.pro")

    >>> ds = read(
    ...     directories=["./snowpack_A/", "./snowpack_B/"],
    ...     time=slice("2024-01-01", "2024-02-01")
    ... )
    """
    pass
```

##### Validating while parsing
The parsers should only throw an error when they cannot read the file. Other validations (such as usage of SI units, physical consistency, etc.) should be validated by a post-processing method.

In case of a corrupt file, the error message should point to the file, location in the file, timestamp that prevents the reading. Possibly, a boolean argument could be added to the parsers that allows the user to `ignore_corrupt_entries` and read the remaining data.

#### Write function signatures
For most `read_*()` functions, there will be an equivalent `.to_*()` method: 
 
 * `.to_crocus()`
 * `.to_netcdf()`
 * `.to_pro()`, `.to_smet()` 
    - If more than one profile in dataset, this will create multiple files. Hence, allow to provide directory name instead of filename.

---
## Data manipulation and basic postprocessing
In addition to xarrays's functionality, several data manipulation methods and convenience wrappers may be required, for example:

### Subsetting and filtering
As explained above, the parsers will allow to subset datasets while reading. xarray provides native functionality to select an even smaller slice of data along one or more dimensions using the `.sel()` method after reading (i.e., subsetting). xarray also provides functionality to mask or drop data based on a condition using the `.where()` method (i.e., filtering). It is therefore not clear at this stage whether any additional functionality is required. Detailed tutorials may be needed.

#### Examples
```python
# Subset ‘ds’ to a single station, a two-week period in January 2024,
# the slope configuration with 38° inclination facing south,
# and the deterministic simulation (we assume there are also others):
subset = ds.sel(
    location="station_A",
    time=slice("2024-01-01", "2024-01-15"),
    slope=(ds.slope_angle == 30) & (ds.azimuth == 180),
    realization=(ds["realization"].labels.get() == "deterministic")
)

# Subset for surface layer in each profile using index subsetting:
surface_index = ds["z"].notnull().argmax("layer")
ds.isel(layer=surface_index)
```

```python
# Filter to only unstable layers & drop stable layers/profiles:
ds_filtered = ds.where(ds["p_unstable"] > 0.77, drop=True)

# Extract the snow height of any profile with an unstable layer:
unstable_profiles = (ds["p_unstable"] > 0.77).any(dim="layer")
hs_unstable = ds["HS"].where(unstable_profiles, drop=True)

# Extract the highest value of p_unstable for each profile with status 2 (i.e., "valid profile with snow")
ds_valid = ds.where((ds["profile_status"] == 2), drop=True)
max_p_unstable = ds_valid["p_unstable"].max(dim="layer")
```

### Concatenating and merging
Both the wrapper function `read()` and the user need to be able to concatenate and merge datasets. Similarly to subsetting and filtering, xarray already implements those functionalities, which we can likely fully adopt. Detailed tutorials may be needed.

Application scenarios:

* Combining simulations from separate locations, times, etc. (i.e., along one or multiple dimensions)

* Fully merging different datasets to update entries (e.g., overwriting of existing data fields from past forecast with current nowcast), applying user-selected merge options.
    - Metadata will be handled by xarray's merge routine *and* can also be changed post-hoc by the user.

#### Examples
```python
# Concatenate two regions
ds_combined = xr.concat([ds_region_A, ds_region_B], dim="location")
```

```python
# Merge two simulations, letting ds_new override ds_old on conflicts
ds_merged = xr.merge([ds_old, ds_new], compat="override", join="inner")

# Many more options using merge(), .update(), or .combine_first()!

```

### Validation
After reading a dataset, the user should be able to verify

 * whether all units are in SI units (see Meteo France's snowprofile package),
 * whether the profiles are physically meaningful and internally consistent (see SNOWPACK CAAML plugin, and potentially Meteo France's snowprofile package?)

### Snow-cover postprocessing

Lastly, to illustrate the power of the gridded data structure, the package will implement a small selection of simple postprocessing routines, such as analytical stability indices and snowpack metrics (e.g., change of 24 hour snow height, slab density, liquid water content index, etc.).

### Tabular data inspection
Users need to be able look at the raw data in convenient ways. We will fully rest on `xarray` and `pandas` to create these tabular views for us.

xarray displays high-level summaries of multi-dimensional datasets and their attributes, so that users understand the dimensions of their selected data subset. For example,

```python
# Subset a full timeseries from one specific location
ds_sub = ds.sel(location="DAVOS", slope=0, realization=0)
print(ds_sub)
# <xarray.Dataset>
# Dimensions:      (location: 1, time: 10, slope: 1, realization: 1, layer: 50)
# Coordinates:
#   * location     (location) object 'DAVOS'
#   * time         (time) datetime64[ns] 2025-01-01 2025-01-02 ... 2025-01-10
#   * slope        (slope) int64 0
#   * realization  (realization) int64 0
#   * layer        (layer) int64 0 1 2 3 4 ... 49

# Data variables:
#     HS           (location, time, slope, realization) float32 ...
#     density      (location, time, slope, realization, layer) float32 ...
#     temperature  (location, time, slope, realization, layer) float32 ...

# Attributes:
#     ...
```

pandas can nicely visualize spreadsheet-like tables for layers and their properties or timeseries of scalars. For example,

```python
# Inspect a layers dataframe
ds_sub = ds.sel(location="DAVOS", time="2025-01-01", slope=0, realization=0)
df = ds_sub.squeeze().to_dataframe()
print(df)
#               z     temperature    ...
# Layer                                                   
# 0           -50.0       -0.20    
# 1           -47.6       -0.95    
# 2           -43.2       -2.80    
# …             …          …       
# 49            0.0      -10.05    
```

### Visualization
Although not a focus of this package, some visualization tools need to be provided to users to ensure they can inspect their data graphically.

#### Profile viewer
To view individual stratigraphy profiles and their timeseries after subsetting a larger dataset to a single profile, two alternatives appear viable:

1. [NiViz](https://niviz.org/) could be harnessed to create interactive plots as being done in the AWSOME framework [here](https://gitlab.com/avalanche-warning/visualization/niviz).

2. [snowpat](https://pypi.org/project/snowpat/)'s plotting functionality could be borrowed. Either snowpat could be enhanced to plot this packages data format (one single profile as `xarray.Dataset`), or this package could facilitate converting one single profile to snowpat's data format.

#### 2D map views
As mentioned earlier, the 1D representation of geographic location does not allow plotting of 2D maps out of the box. Since the 2D information is provided through coordinate variables, scalar data variables can be easily reshaped for regular spatial configurations. In the case of irregular polygonal domains (e.g., variable-resolution meshes), user-provided geometry files (e.g., GeoJSON, Shapefile) are required. This package will implement the reshaping routines, so that users can plot their maps using geopandas, xarray, or other libraries in a straightforward way.




