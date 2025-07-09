# Specifying postprocessing extensions to the base class

## Contents

 1. [Context](#context)
     1. Objectives
     2. Requirements
     3. Application scenarios
 2. [Application scenario 1: Adding new data variables without altering the existing data structure](#application-scenario-1-adding-new-data-variables-without-altering-the-existing-data-structure)
    1. Adding methods directly into the xsnowDataset class
    2. Providing custom functions in separate modules or packages
 3. [Application scenario 2: New data streams and functionality](#application-scenario-2-new-data-streams-and-functionality)
    1. Extensions as decorator classes
        1. Benefits
        2. Disadvantages
        3. Demo implementation
    2. Desired extensions
        1. Hazard chart data
        2. Summary of bulk snowcover instability
        3. Simplified stratigraphy
 4. [Open questions, prompts, uncertainties](#open-questions-prompts-uncertainties)

## Context
### Objectives

 * Extend the base data class to support postprocessed data from large gridded model datasets for use in operational products
 * Develop a generalized data model that shares dimensions with snow profile data but includes key features tailored to postprocessed outputs
 * Provide a flexible, consistent structure for computing, storing, and accessing postprocessed data


### Requirements
 * Allow for customization: every stakeholder (research group, warning agency) may have their own needs
 * Allow for development outside this working group. Anybody should be able to implement extensions on their platforms (if desired) and share their extension with the user community
 * Avoid having to deal with unused extensions: Agency A may need an extension that is not relevant for agency B. Agency B should therefore not have to deal with code related to that extension.


### Application scenarios
 1. Adding new data variables that perfectly fit into the existing xsnowDataset structure, such as
    * Scalars: HN24, HN72, HST, wind runs, etc
    * Layer properties: p_unstable, etc.

 2. Adding new data streams and functionality beyond the existing xsnowDataset structure, such as
    * hazard chart data

 3. Altering the existing base dimensions before applying scenarios 1 or 2, such as
    * aggregating existing locations into polygons before
    * storing regular scalars and/or aggregate layers

Application scenarios 1 & 2 seem most common. Will we need scenario 3? Do we need to store data for scenario 3, or will those views be generated on-the-fly from the non-aggregated data?


[[Back to table of contents](#contents)]


## Application scenario 1: Adding new data variables without altering the existing data structure
This seems like a rather straightforward scenario with two obvious solutions:

### Adding methods directly into the xsnowDataset class
 * Very common and simple-to-compute measures can be directly added as methods to the existing base class.
 * Examples: HN24, wind runs, etc
 * Usage:
```python
ds = xsnow.read("file.nc")
ds = ds.compute_HN24()
```

### Providing custom functions in separate modules or packages
 * Less common measures, ML models, etc can be computed through functions provided outside the xsnow package
 * Example implementation of `add_HN()` in a separate module:
    - Plain function in custom module
    - Immutability: return a new object, do not alter the existing object in-place
    - Metadata preservation: Units flow automatically
    - Type-safe: make the function compatible with more complex extensions outlined in the next Section.
```python
from typing import TypeVar
import xarray as xr

# Let xsnowDataset_variant be any subclass of xsnowDataset
xsnowDataset_variant = TypeVar("T", bound="xsnowDataset")

def add_HN(ds: xsnowDataset_variant, window: int = 24) -> xsnowDataset_variant:
    """
    Compute the n-period snow-height change and attach it
    as a new scalar variable `HN{window}` with the same
    dims as `HS` (location, time, realization, slope).
    """
    # 1) compute
    dh = ds.ds["HS"].diff("time", n_periods=window)

    # 2) name & metadata
    name = f"HN{window}"
    attrs = {
        "long_name": f"{window}-hour snow height change",
        "units": ds.ds["HS"].attrs.get("units", ""),
    }

    # 3) assign to a new xarray.Dataset
    ds2_xr: xr.Dataset = ds.ds.assign({ name: dh })
    ds2_xr[name].attrs.update(attrs)

    # 4) wrap back in the same class
    return ds.__class__(ds2_xr)
```
 * Usage
```python
import xsnow
from my_custom_extensions import add_HN

ds = xsnow.read("file.nc")
ds = add_HN(ds, window=24)
ds = add_HN(ds, window=72)
```

[[Back to table of contents](#contents)]

## Application scenario 2: New data streams and functionality
### Extensions as decorator classes
  * Each postprocessing functionality and resulting data stream is implemented as separate extension.
  * Each extension defines their own **decorator class** that wraps the base xsnowDataset class and
    1. copies the base dataset
    2. extends that copy with an additional dimension and associated data variables
    3. exposes the same interface, so that postprocessing extensions can be chained in any order, and the existing functionality remains plus new functionality contained in each extension is added.
  * Key will be to give the decorated object exactly the same surface API as the base object, while still exposing extra methods once it’s been wrapped. To do so, the decorator will have to proxy all attribute and method lookups to its internal `.ds`, and—if the result of a method call is a new xarray.Dataset—re-wrap it in the same decorator class.
    
    
#### Benefits 
  * Efficient: no duplication of base dimensions needed; proper build-up from ground upwards
  * Non-intrusive: no extension ever touches or modifies the original data model 
  * Modular: each class only knows about its own dimension and functionality
  * Lean user-interface: each user only has to deal with dimensions and functionality that they are interested in. Unused extensions are completely hidden.
  * Distributed: each stakeholder can implement their own extensions and make them public if they choose
  * Extensible: future postprocessing methods and data streams can each become their own extension

#### Disadvantages
  * Slightly more boilerplate: write a new class for every postprocessing feature
  * There is a learning curve to understanding decorator classes and how to implement them
    
#### Demo implementation

##### Base code
```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 1. For reference: The base data model (defined in the xsnow base package):
class xsnowDataset:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds
    
    def __getattr__(self, name):
        # forward everything else to the underlying xr.Dataset
        return getattr(self.ds, name)
    
    # … other existing methods …

# 2. A generic class for all “decorators”
class DatasetDecorator:
    def __init__(self, base: xsnowDataset):
        self.base = base
        self.ds   = base.ds
        # stash __init__ args for later re‐wrapping
        self._decorator_args = getattr(base, "_decorator_args", (base,))
        self._decorator_kwargs = getattr(base, "_decorator_kwargs", {})

    def __getattr__(self, name):
        attr = getattr(self.ds, name)
        if callable(attr):
            def method(*args, **kwargs):
                result = attr(*args, **kwargs)
                # if it returns a new Dataset, re‐wrap it
                if isinstance(result, xr.Dataset):
                    cls = self.__class__
                    # re‐instantiate with same init args
                    return cls(*self._decorator_args, **self._decorator_kwargs, ds=result)
                else:
                    return result
            return method
        else:
            return attr
```

##### Extension
```python
# 3. HazardChartData decorator
class HazardChartData(DatasetDecorator):
    """
    Decorator that adds a 'hazardChartElement' dimension to an xsnowDataset
    and computes per‐element values, depths, graincodes, etc.
    """
    def __init__(self,
                 base: xsnowDataset,
                 ds: xr.Dataset = None):
        # allow 2 forms: (base) or (base, ds=already_built_ds)
        if ds is None:
            super().__init__(base)
            self._decorator_args   = (base, elements, problem_types, stability_indices)
            self._decorator_kwargs = {}
            self.ds = self._compute_hazardChartElements()
        else:
            # re‐wrapping path
            self.base = base
            self.ds   = ds
            self._decorator_args   = (base, elements, problem_types, stability_indices)
            self._decorator_kwargs = {}

    def _compute_hazardChartElements(self) -> xr.Dataset:
        ds0 = self.ds
        dims = ("location","time","slope","realization","hazardChartElement")
        ne = 100  # number of resulting hazardChartElements (init with fixed/user-provided value)
        shape = (*[ds0.sizes[d] for d in dims[:-1]], ne)

        # allocate placeholders
        vals   = np.full(shape, np.nan, dtype=float)
        depths = np.full(shape, np.nan, dtype=float)
        grains = np.zeros(shape,       dtype=int)
        inst   = np.full(shape, "",    dtype=object)
        depthb = np.full(shape, "",    dtype=object)

        # --- here: compute real data arrays ---
        # for i, elem in enumerate(self.elements):
        #     vals[...,i]   = ...
        #     depths[...,i] = ...
        #     grains[...,i] = ...
        #     inst[...,i]   = ...
        #     depthb[...,i] = ...

        hz = xr.Dataset(
            data_vars={
                "hazard_value":          (dims, vals),
                "hazard_depth":          (dims, depths),
                "hazard_graincode":      (dims, grains),
                "instabilityDrivenBy":   (dims, inst),
                "depthDrivenBy":         (dims, depthb),
            },
            coords={
                "hazardChartElement": ("hazardChartElement", ne)
            }
        )
        return ds0.merge(hz)

    def plot_hazardChart(self,
                         problemType: str|None=None,
                         stabilityIndex: str|None=None,
                         **plt_kwargs):
        ds = self.ds

        # filter by problemType / stabilityIndex
        if problemType is not None:
            mask = ds["problemType"] == problemType
            ds   = ds.sel(hazardChartElement=mask)

        if stabilityIndex is not None:
            mask = ds["stabilityIndex"] == stabilityIndex
            ds   = ds.sel(hazardChartElement=mask)

        hv = ds["hazard_value"]
        hd = ds["hazard_depth"]
        
        plt.scatter(hd.values, hv.values, **plt_kwargs)
        plt.xlabel("Depth")
        plt.ylabel("Value")
        plt.title("Hazard Chart)
        plt.show()
```

##### Usage
```python
import xsnow
import xsnow_HCD_extension as xsnow_hcd
ds = xsnow.read("file.nc")
ds_ext  = xsnow_hcd.HazardChartData(ds)

# Subset to a single location/time/slope/realization first:
ds_sub = ds_ext.sel(location=1, slope=0, realization=0,time="2025-01-01")

# Convert to DataFrame (layer dim is automatically dropped because hazard vars don't use it):
df = ds_sub[["hazard_value","hazard_depth","problemType","stabilityIndex"]].squeeze().to_dataframe()
print(df)
#                       problemType stabilityIndex  value   depth   graincode  instabilityDrivenBy 
# hazardChartElement                                                   
# 0                     persistent      sk38_rta     0.80    -50.0      123       "2024-01-17"
# 1                     persistent      punstable    0.87    -47.6      123       "2024-01-17"
# …                     …          …       
# 23                    storm           punstable    0.78    -18.5      224       "new_snow"
## potential additional columns (omitted above): graincode_sec, depthDrivenBy
```
```python
# Plot by element name:
ds_sub.plot_scatter(stabilityIndex="sk38_rta", problemType="persistent")
```

### Desired extensions
#### Hazard chart data
 * see Demo implementation above, specifically the Usage section with the tabular data view per single profile
 * part of `qmah` in a json variant (called `hazardChartElement`, [demo link](https://gitlab.com/avalanche-warning/visualization/albina-dashboard/-/issues/3#note_2397368932))

#### Summary of bulk snowcover instability
 * part of `qmah` in a json variant (called `snp_characteristics`)
 * extracts weakest instabilities per stability index
 * in our dataset environment, it could even be enough to point to the specific `layer`
 * if actually extracting the data, it could look like this in a dataframe view:
```python
#                       stabilityIndex  value   depth   graincode       
# stabilitySummary                                                   
# 0                     sk38_rta        0.80    -50.0      123       
# 1                     punstable       0.87    -47.6      123       
# …                     …          …       
## potential additional columns (omitted above): graincode_sec, size
```

#### Simplified stratigraphy
 * "regular" `layer` dimension, but with fewer layers: only one layer per datetag
 * required for 
    - subsetting hazard chart to specific layers of interest (e.g., "Jan 17")
    - hazardChartElement columns "instabilityDrivenBy" & "depthDrivenBy"


[[Back to table of contents](#contents)]

## Open questions, prompts, uncertainties
* Data variables of newly added dimensions need to adhere to a naming convention to not create duplicates of variables names in the dataset.
    - For example, 'value', 'depth' are poor choices, because it is not clear that they refer to the hazardChartElement dimension. 
    - Maybe prepend names with an extension abbreviation, such as 'hcd_value', 'hcd_depth', etc.


[[Back to table of contents](#contents)]