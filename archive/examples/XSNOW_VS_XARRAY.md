# xsnow.read() vs xr.open_dataset() - Quick Reference

**When to use which function for reading data**

---

## Quick Answer

```python
# Reading SNOWPACK source files (.pro/.smet):
import xsnow
ds = xsnow.read('file.pro')  ✅

# Reading our archives (.nc):
import xarray as xr
ds = xr.open_dataset('operational.nc')  ✅
```

---

## Detailed Explanation

### xsnow.read() - For SNOWPACK Source Files

**Purpose:** Parse raw SNOWPACK output files

**Use for:**
- .pro files (SNOWPACK profiles)
- .smet files (SNOWPACK met data)
- Raw SNOWPACK model output

**What it does:**
1. Parses SNOWPACK-specific file format
2. Extracts metadata (headers, settings)
3. Converts to xarray Dataset
4. Handles SNOWPACK data structures

**Example:**
```python
import xsnow

# Single file
ds = xsnow.read('/data/snowpack/point_001/aspect_000.pro')

# Multiple files
files = [
    '/data/snowpack/point_001/aspect_000.pro',
    '/data/snowpack/point_001/aspect_090.pro',
    '/data/snowpack/point_002/aspect_000.pro'
]
ds = xsnow.read(files, lazy=True)

# This is what operational_archive.py does internally!
```

**File format it reads:**
```
[HEADER]
station_id = VIR1A
station_name = Point 001
...

[DATA]
0500,19950101,0.00,273.15,0.00,...
0501,19950101,0.01,273.14,0.00,...
...
```

---

### xr.open_dataset() - For NetCDF Archives

**Purpose:** Read standard NetCDF files

**Use for:**
- operational.nc (our operational archive)
- archive_*.nc (our research archives)
- Any NetCDF file (standard format)

**What it does:**
1. Opens NetCDF file (standard format)
2. Loads xarray Dataset directly
3. No parsing needed (already in xarray format)

**Example:**
```python
import xarray as xr

# Operational archive (we created this from SNOWPACK files)
ds = xr.open_dataset('/data/operational.nc')

# Research archive
ds = xr.open_dataset('/data/archives/archive_2024_12.nc')

# This is what YOU use for forecasting/analysis!
```

**File format it reads:**
```
NetCDF format (binary, efficient)
- Already an xarray Dataset
- Coordinates, dimensions, variables
- Metadata, attributes
- Optimized for reading
```

---

## The Complete Workflow

```
SNOWPACK Output Files (.pro/.smet)
         ↓
    xsnow.read()  ← operational_archive.py uses this
         ↓
    xsnow.Dataset (in memory)
         ↓
    .to_netcdf()  ← Saves underlying xarray.Dataset
         ↓
NetCDF Archive (.nc files)
         ↓
  xr.open_dataset()  ← YOU use this for querying
         ↓
    xarray.Dataset (most operations work here)
         ↓
  xsnow.Dataset(ds)  ← Optional: only if you need xsnow methods
         ↓
    xsnow.Dataset (SNOWPACK-specific functionality)
```

**Two paths for analysis:**

**Path 1: Plain xarray (90% of use cases)**
```python
import xarray as xr

ds = xr.open_dataset('operational.nc')  # xarray.Dataset
latest = ds.HS.isel(time=-1).compute()  # Works! ✅
max_hs = ds.HS.max(dim='time')  # Works! ✅
```

**Path 2: Convert to xsnow (special analysis)**
```python
import xarray as xr
import xsnow

ds_xr = xr.open_dataset('operational.nc')  # xarray.Dataset
ds_xs = xsnow.Dataset(ds_xr)  # xsnow.Dataset
ds_xs.plot_profile(location='VIR1A')  # xsnow-specific! ✅
```

---

## Converting Between Types

### From xarray.Dataset to xsnow.Dataset

```python
import xarray as xr
import xsnow

# Load NetCDF archive (returns xarray.Dataset)
ds_xarray = xr.open_dataset('/data/operational.nc')

# Convert to xsnow.Dataset for xsnow-specific methods
ds_xsnow = xsnow.Dataset(ds_xarray)

# Now you have access to xsnow methods:
ds_xsnow.plot_profile(location='VIR1A')
ds_xsnow.get_layer_properties()
# etc.
```

### When to Convert

**Don't convert (most common):**
```python
import xarray as xr

ds = xr.open_dataset('/data/operational.nc')

# Standard operations work fine with plain xarray:
latest = ds.HS.isel(time=-1).compute()  ✅
max_hs = ds.HS.max(dim='time')  ✅
location_data = ds.sel(location='VIR1A')  ✅

# This is sufficient for 90% of use cases!
```

**Do convert when you need:**
```python
import xarray as xr
import xsnow

ds_xarray = xr.open_dataset('/data/operational.nc')
ds_xsnow = xsnow.Dataset(ds_xarray)

# xsnow-specific functionality:
ds_xsnow.plot_profile()  # SNOWPACK-specific plots
ds_xsnow.get_weak_layers()  # Layer analysis
ds_xsnow.stability_indices()  # Stability calculations
```

---

## Common Mistakes

### ❌ Wrong: Using xsnow.read() for archives

```python
import xsnow

# This will FAIL:
ds = xsnow.read('/data/operational.nc')
# Error: Not a SNOWPACK .pro/.smet file!
```

### ❌ Wrong: Using xr.open_dataset() for .pro files

```python
import xarray as xr

# This will FAIL or give garbage:
ds = xr.open_dataset('/data/snowpack/point_001/aspect_000.pro')
# Error: Not a NetCDF file!
```

### ✅ Correct: Match tool to file type

```python
import xsnow
import xarray as xr

# For SNOWPACK source files:
ds_source = xsnow.read('/data/snowpack/point_001/aspect_000.pro')  ✅

# For our archives:
ds_archive = xr.open_dataset('/data/operational.nc')  ✅
```

---

## File Type Quick Reference

| File Extension | Location | Use | Tool |
|---------------|----------|-----|------|
| .pro | /data/snowpack/... | SNOWPACK output | xsnow.read() |
| .smet | /data/snowpack/... | SNOWPACK met data | xsnow.read() |
| operational.nc | /data/ | Our operational archive | xr.open_dataset() |
| archive_*.nc | /data/archives/ | Our research archives | xr.open_dataset() |

---

## Why This Matters

### Performance

```python
# Slow: Re-parsing SNOWPACK files every time
for analysis in daily_forecasts:
    ds = xsnow.read(all_pro_files)  # Parse 114,300 files!
    latest = ds.HS.isel(time=-1)
    # Takes hours ❌

# Fast: Query pre-processed archive
for analysis in daily_forecasts:
    ds = xr.open_dataset('operational.nc')  # Load once!
    latest = ds.HS.isel(time=-1)
    # Takes seconds ✅
```

### Efficiency

```python
# Inefficient: Parse raw files
files = list_all_pro_files()  # 114,300 files
ds = xsnow.read(files)  # Parse all
latest = ds.HS.isel(time=-1)
# Reads 173 GB, processes 114,300 files ❌

# Efficient: Load archive
ds = xr.open_dataset('operational.nc')  # 500 MB
latest = ds.HS.isel(time=-1)
# Reads 500 MB, instant access ✅
```

---

## In the Context of This System

### What operational_archive.py Does (Internal)

```python
# Step 1: Read SNOWPACK source files
changed_files = find_changed_pro_files()
ds = xsnow.read(changed_files)  # Parse SNOWPACK format

# Step 2: Process
ds = apply_time_window(ds)

# Step 3: Save as NetCDF
ds.to_netcdf('operational.nc')
```

### What YOU Do (For Forecasting)

```python
# Step 1: Load processed archive
ds = xr.open_dataset('operational.nc')  # Pre-processed!

# Step 2: Analyze
latest = ds.HS.isel(time=-1).compute()
max_hs = ds.HS.max(dim='time').compute()

# Fast! No parsing needed ✅
```

---

## Multiple Files

### For SNOWPACK Source Files

```python
import xsnow

# Multiple .pro files
files = glob.glob('/data/snowpack/**/*.pro')
ds = xsnow.read(files, lazy=True)
```

### For NetCDF Archives

```python
import xarray as xr

# Multiple .nc files (e.g., monthly archives)
ds = xr.open_mfdataset('/data/archives/archive_2024_*.nc')
# Opens: archive_2024_01.nc, archive_2024_02.nc, etc.
```

---

## Summary

**Simple rule:**
```
Reading SNOWPACK .pro/.smet files? → xsnow.read()
Reading our .nc archives? → xr.open_dataset()
Need xsnow-specific methods? → xsnow.Dataset(ds)
```

**Functionality comparison:**

| Operation | xarray.Dataset | xsnow.Dataset |
|-----------|---------------|---------------|
| Load .nc file | ✅ xr.open_dataset() | Via conversion |
| Load .pro file | ❌ | ✅ xsnow.read() |
| Select time/location | ✅ .sel(), .isel() | ✅ Same |
| Aggregations (max, mean) | ✅ .max(), .mean() | ✅ Same |
| Plotting (basic) | ✅ .plot() | ✅ Same |
| SNOWPACK plots | ❌ | ✅ .plot_profile() |
| Weak layer analysis | ❌ | ✅ .get_weak_layers() |
| Stability indices | ❌ | ✅ Custom methods |

**When to use which:**
- **xarray.Dataset:** 90% of forecasting/analysis (fast, simple)
- **xsnow.Dataset:** SNOWPACK-specific analysis (when needed)

**Why we use archives:**
- SNOWPACK files: Large, slow to parse (173 GB raw)
- NetCDF archives: Small, fast to load (500 MB processed)
- Query time: Hours vs seconds

**In practice:**
- operational_archive.py: Uses xsnow.read() to process source files
- You (forecasters/researchers): Use xr.open_dataset() to query archives
- Convert to xsnow.Dataset only when you need SNOWPACK-specific methods

**Bottom line:**
Archives exist so you DON'T have to parse 114,300 SNOWPACK files every time you want current data! 🎯
