# Usage Examples - Complete Guide

**Practical examples for querying operational and research archives**

---

## Quick Start

### Basic Query (Most Common)

```python
import xarray as xr

# Load operational archive
ds = xr.open_dataset('/data/operational.nc')

# Get latest snow depth
latest_hs = ds.HS.isel(time=-1).compute()

print(f"Latest data: {ds.time.values[-1]}")
print(f"Max snow depth: {latest_hs.max().values:.2f} m")
```

---

## Operational Forecasting Examples

### Example 1: Latest Conditions for All Locations

```python
import xarray as xr
import numpy as np

# Load operational archive
ds = xr.open_dataset('/data/operational.nc')

# Get latest timestep
latest = ds.isel(time=-1)

# Extract key variables
hs = latest.HS.compute()  # Snow depth
ts = latest.TS.compute()  # Surface temperature
lwc = latest.LWC.compute()  # Liquid water content

# Find dangerous conditions
deep_snow = hs > 3.0  # Locations with >3m snow
warm_surface = ts > 0  # Above freezing
wet_snow = lwc > 0.03  # High water content

# Combine conditions
alert_zones = deep_snow & warm_surface & wet_snow

# Get location IDs with alerts
alert_locs = ds.location.values[alert_zones.values]

print(f"Alert zones ({len(alert_locs)}):")
for loc in alert_locs[:10]:  # First 10
    print(f"  {loc}: HS={hs.sel(location=loc).values:.2f}m, "
          f"TS={ts.sel(location=loc).values:.1f}°C, "
          f"LWC={lwc.sel(location=loc).values:.3f}")
```

### Example 2: Time Series for Specific Location

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load operational archive
ds = xr.open_dataset('/data/operational.nc')

# Select location
location = ds.sel(location='VIR1A')

# Get time series
hs_timeseries = location.HS.compute()
ts_timeseries = location.TS.compute()

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Snow depth
ax1.plot(location.time, hs_timeseries)
ax1.set_ylabel('Snow Depth (m)')
ax1.set_title(f'Location: VIR1A - Last 14 Days')
ax1.grid(True)

# Surface temperature
ax2.plot(location.time, ts_timeseries)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.set_ylabel('Surface Temperature (°C)')
ax2.set_xlabel('Date')
ax2.grid(True)

plt.tight_layout()
plt.savefig('location_timeseries.png')
print("Saved: location_timeseries.png")
```

### Example 3: Daily Report Generation

```python
import xarray as xr
from datetime import datetime
import pandas as pd

# Load operational archive
ds = xr.open_dataset('/data/operational.nc')

# Get latest timestep
latest = ds.isel(time=-1)

# Compute statistics
stats = {
    'timestamp': pd.Timestamp(latest.time.values),
    'n_locations': len(ds.location),
    'max_hs': float(latest.HS.max().values),
    'mean_hs': float(latest.HS.mean().values),
    'max_ts': float(latest.TS.max().values),
    'min_ts': float(latest.TS.min().values),
    'locations_above_2m': int((latest.HS > 2.0).sum().values),
    'locations_wet': int((latest.LWC > 0.03).sum().values),
}

# Generate report
print("="*60)
print(f"SNOWPACK Daily Report - {stats['timestamp']}")
print("="*60)
print(f"Locations monitored: {stats['n_locations']:,}")
print(f"Snow depth (max/mean): {stats['max_hs']:.2f}m / {stats['mean_hs']:.2f}m")
print(f"Surface temp range: {stats['min_ts']:.1f}°C to {stats['max_ts']:.1f}°C")
print(f"Deep snow (>2m): {stats['locations_above_2m']:,} locations")
print(f"Wet snow conditions: {stats['locations_wet']:,} locations")
print("="*60)

# Save to file
with open(f'daily_report_{datetime.now():%Y%m%d}.txt', 'w') as f:
    f.write(f"SNOWPACK Daily Report - {stats['timestamp']}\n")
    f.write(f"Locations: {stats['n_locations']:,}\n")
    f.write(f"Max HS: {stats['max_hs']:.2f}m\n")
    # ... etc
```

### Example 4: Change Detection (24-hour)

```python
import xarray as xr

# Load operational archive
ds = xr.open_dataset('/data/operational.nc')

# Get current and 24h ago (assuming hourly data)
current = ds.isel(time=-1)
past_24h = ds.isel(time=-24)

# Calculate changes
hs_change = current.HS - past_24h.HS
ts_change = current.TS - past_24h.TS

# Find significant changes
rapid_accumulation = hs_change > 0.5  # >50cm in 24h
rapid_warming = ts_change > 5.0  # >5°C warming

# Compute
hs_change_vals = hs_change.compute()
rapid_acc_locs = ds.location.values[rapid_accumulation.compute().values]

print(f"Locations with rapid accumulation (>50cm/24h): {len(rapid_acc_locs)}")
for loc in rapid_acc_locs[:10]:
    change = hs_change_vals.sel(location=loc).values
    print(f"  {loc}: +{change*100:.1f} cm")
```

---

## Using xsnow.Dataset for Advanced Analysis

### Example 5: Converting for xsnow-Specific Methods

```python
import xarray as xr
import xsnow

# Load as xarray
ds_xarray = xr.open_dataset('/data/operational.nc')

# Convert to xsnow.Dataset
ds = xsnow.Dataset(ds_xarray)

# Now you can use xsnow-specific methods
# (availability depends on your xsnow version)

# Profile plotting
ds.plot_profile(location='VIR1A', time='2024-12-20T12:00:00')

# Weak layer detection (if implemented)
# weak_layers = ds.get_weak_layers(location='VIR1A')

# Custom analysis
# stability = ds.compute_stability_indices()
```

### Example 6: Layer-by-Layer Analysis (if xsnow supports)

```python
import xarray as xr
import xsnow

ds_xarray = xr.open_dataset('/data/operational.nc')
ds = xsnow.Dataset(ds_xarray)

# Select location and time
profile = ds.sel(location='VIR1A', time='2024-12-20T12:00:00')

# Access layer properties (if available in xsnow)
# This depends on what xsnow exposes after reading from NetCDF
# Example conceptual code:

# if hasattr(ds, 'get_layer_properties'):
#     layers = ds.get_layer_properties(location='VIR1A')
#     for layer in layers:
#         print(f"Depth: {layer.depth}m, Type: {layer.type}, "
#               f"Hardness: {layer.hardness}")
```

---

## Research Examples

### Example 7: Seasonal Analysis

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load full year archive
ds = xr.open_dataset('/data/archives/archive_2024.nc')

# Select winter season (Dec-Mar)
winter = ds.sel(time=slice('2024-12-01', '2025-03-31'))

# Compute seasonal statistics
winter_max_hs = winter.HS.max(dim='time').compute()
winter_mean_hs = winter.HS.mean(dim='time').compute()

# Find peak snow depth date for each location
peak_dates = winter.HS.argmax(dim='time')

# Create map of maximum snow depths
plt.figure(figsize=(12, 8))
plt.scatter(ds.lon, ds.lat, c=winter_max_hs, cmap='Blues', s=10)
plt.colorbar(label='Max Snow Depth (m)')
plt.title('Winter 2024-25 Maximum Snow Depth')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('winter_max_hs_map.png')
```

### Example 8: Monthly Climatology

```python
import xarray as xr
import pandas as pd

# Load full year
ds = xr.open_dataset('/data/archives/archive_2024.nc')

# Group by month
monthly = ds.HS.groupby('time.month')

# Compute monthly statistics
monthly_mean = monthly.mean(dim='time').compute()
monthly_max = monthly.max(dim='time').compute()
monthly_std = monthly.std(dim='time').compute()

# Create summary table
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for i, month in enumerate(months, 1):
    if i in monthly_mean.month.values:
        mean = monthly_mean.sel(month=i).mean().values
        max_val = monthly_max.sel(month=i).mean().values
        std = monthly_std.sel(month=i).mean().values
        print(f"{month}: Mean={mean:.2f}m, Max={max_val:.2f}m, Std={std:.2f}m")
```

### Example 9: Multi-Year Comparison

```python
import xarray as xr

# Load multiple years
ds_2023 = xr.open_dataset('/data/archives/archive_2023.nc')
ds_2024 = xr.open_dataset('/data/archives/archive_2024.nc')

# Extract winter periods
winter_2023 = ds_2023.sel(time=slice('2023-12-01', '2024-03-31'))
winter_2024 = ds_2024.sel(time=slice('2024-12-01', '2025-03-31'))

# Compare maximum snow depths
max_2023 = winter_2023.HS.max(dim='time').compute()
max_2024 = winter_2024.HS.max(dim='time').compute()

# Calculate difference
difference = max_2024 - max_2023

# Statistics
print(f"Winter 2023-24 mean max HS: {max_2023.mean().values:.2f}m")
print(f"Winter 2024-25 mean max HS: {max_2024.mean().values:.2f}m")
print(f"Average change: {difference.mean().values:.2f}m")
print(f"Locations with deeper snow: {(difference > 0).sum().values}")
print(f"Locations with less snow: {(difference < 0).sum().values}")
```

### Example 10: Export to Other Formats

```python
import xarray as xr
import pandas as pd

# Load archive
ds = xr.open_dataset('/data/operational.nc')

# Convert to pandas DataFrame for specific location
location_data = ds.sel(location='VIR1A').to_dataframe()

# Save as CSV
location_data.to_csv('VIR1A_timeseries.csv')

# Or export latest timestep for all locations as CSV
latest = ds.isel(time=-1).to_dataframe()
latest.to_csv('latest_conditions_all_locations.csv')

print("Exported to CSV files")
```

---

## Advanced Queries

### Example 11: Spatial Aggregation by Region

```python
import xarray as xr
import numpy as np

# Load operational archive
ds = xr.open_dataset('/data/operational.nc')

# Define regions (example - adjust to your coordinate system)
def classify_region(lat, lon):
    if lon < -106.5:
        return 'West'
    elif lon > -105.5:
        return 'East'
    else:
        return 'Central'

# Create region labels (vectorized)
regions = np.array([classify_region(lat, lon) 
                    for lat, lon in zip(ds.lat.values, ds.lon.values)])

# Add as coordinate
ds = ds.assign_coords(region=('location', regions))

# Group by region
regional = ds.groupby('region')

# Compute regional statistics
latest = ds.isel(time=-1)
regional_stats = {
    region: {
        'mean_hs': float(latest.sel(location=latest.region == region).HS.mean().values),
        'max_hs': float(latest.sel(location=latest.region == region).HS.max().values),
        'n_locations': int((latest.region == region).sum().values),
    }
    for region in np.unique(regions)
}

for region, stats in regional_stats.items():
    print(f"{region}: {stats['n_locations']} locations, "
          f"mean={stats['mean_hs']:.2f}m, max={stats['max_hs']:.2f}m")
```

### Example 12: Percentile Analysis

```python
import xarray as xr

# Load archive
ds = xr.open_dataset('/data/archives/archive_2024.nc')

# Compute percentiles across time
p10 = ds.HS.quantile(0.10, dim='time').compute()
p50 = ds.HS.quantile(0.50, dim='time').compute()
p90 = ds.HS.quantile(0.90, dim='time').compute()

# Find locations that frequently have deep snow
frequently_deep = (p50 > 2.0)

print(f"Locations with median HS > 2m: {frequently_deep.sum().values}")

# Compare current to percentiles
current = ds.HS.isel(time=-1).compute()
above_p90 = current > p90

print(f"Locations currently above 90th percentile: {above_p90.sum().values}")
```

---

## Time Window Utility Examples

### Example 13: Creating Time-Limited Archives

```bash
# Extract last 30 days from large archive
python apply_time_window.py /data/archives/archive_2024.nc --days 30 --output recent_month.nc

# Create quarterly archive
python apply_time_window.py /data/archives/archive_2024.nc --days 90 --output q4_2024.nc

# Extract last week for quick analysis
python apply_time_window.py /data/operational.nc --days 7 --output last_week.nc
```

### Example 14: Using Time-Limited Archives

```python
import xarray as xr

# Create 7-day archive first (much faster to query)
# Run: python apply_time_window.py operational.nc --days 7 --output week.nc

# Load small archive
ds = xr.open_dataset('week.nc')  # Fast!

# Quick analysis on recent data
daily_mean = ds.HS.resample(time='1D').mean()
daily_max = ds.HS.resample(time='1D').max()

print("Last 7 days summary:")
for time, mean, max_val in zip(daily_mean.time.values,
                                daily_mean.mean(dim='location').values,
                                daily_max.mean(dim='location').values):
    print(f"{time}: mean={mean:.2f}m, max={max_val:.2f}m")
```

### Example 15: Programmatic Time Window Application

```python
import xarray as xr
from datetime import datetime, timedelta

# Load full archive
ds = xr.open_dataset('/data/archives/archive_2024.nc')

# Apply custom time window (last 60 days)
cutoff = datetime.now() - timedelta(days=60)
ds_windowed = ds.sel(time=slice(cutoff.isoformat(), None))

# Save if needed
ds_windowed.to_netcdf('last_60_days.nc')

print(f"Original timesteps: {len(ds.time)}")
print(f"Windowed timesteps: {len(ds_windowed.time)}")
print(f"Size reduction: {(1 - len(ds_windowed.time)/len(ds.time))*100:.1f}%")
```

---

## Best Practices

### Memory Management

```python
import xarray as xr

# For large computations, use chunks
ds = xr.open_dataset('/data/archives/archive_2024.nc', chunks={'time': 100})

# Compute only what you need
result = ds.HS.max(dim='time')  # Lazy
result_computed = result.compute()  # Only compute when needed

# Close datasets when done
ds.close()
```

### Efficient Querying

```python
import xarray as xr

# Select data before computing
ds = xr.open_dataset('/data/operational.nc')

# Bad - loads everything:
# all_data = ds.compute()
# location = all_data.sel(location='VIR1A')

# Good - select first, then compute:
location = ds.sel(location='VIR1A')
location_computed = location.compute()
```

### Handling Missing Data

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('/data/operational.nc')

# Check for NaN values
has_nan = ds.HS.isnull().any()

if has_nan.compute().values:
    print("Warning: Dataset contains missing values")
    
    # Fill with interpolation
    ds_filled = ds.HS.interpolate_na(dim='time', method='linear')
    
    # Or drop locations with missing data
    ds_clean = ds.dropna(dim='location', how='any')
```

---

## Summary

**For daily operations:**
- Use `xr.open_dataset()` for quick queries
- Convert to `xsnow.Dataset()` only for SNOWPACK-specific methods
- Use `apply_time_window.py` to create smaller archives for faster analysis

**For research:**
- Load full archives with `xr.open_dataset()`
- Use xarray's powerful selection and aggregation methods
- Export to other formats (CSV, pandas) as needed

**Key principle:**
Work with the smallest dataset that meets your needs - use time windows, location selection, and variable selection to reduce memory usage and improve performance. 🎯
