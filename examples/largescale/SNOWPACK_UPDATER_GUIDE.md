# SnowpackUpdater - Usage Guide

## Overview

`SnowpackUpdater` is a single, intelligent class that automatically handles both:
- **Initial load**: Creates full historical archive (first run)
- **Updates**: Applies rolling time window (subsequent runs)

**You use the same command every time!** The class auto-detects what to do.

## Quick Start

### Basic Usage (Recommended)

```bash
# Run this same command every time:
python snowpack_updater.py /home/snowpack/runs/output.2024

# First run: Auto-detects no archive, keeps ALL history
# Later runs: Auto-detects archive exists, applies 90-day window
```

### Custom Archive Location

```bash
python snowpack_updater.py /home/snowpack/runs/output.2024 \
    --archive /path/to/snowpack_operational.nc \
    --window-days 90
```

### From Python

```python
from snowpack_updater import SnowpackUpdater

# Same code works for both first run and updates!
updater = SnowpackUpdater(
    data_dir='/home/snowpack/runs/output.2024',
    archive_file='snowpack_operational.nc',
    window_days=90
)

ds = updater.run()
```

## How Auto-Detection Works

```python
if archive_file exists:
    mode = 'update'
    window_days = 90  # Apply rolling window
else:
    mode = 'initial'
    window_days = 0   # Keep all history
```

## Typical Workflow

### First Run (Initial)

```bash
$ python snowpack_updater.py /home/snowpack/runs/output.2024

======================================================================
SNOWPACK ARCHIVE UPDATER
======================================================================
Data directory: /home/snowpack/runs/output.2024
Archive file: snowpack_operational.nc

🔍 Mode: INITIAL
   First run detected - will create full historical archive
   Keeping ALL historical data (no time window)
======================================================================

📊 Scanning for files...
   .pro files:  57,150
   .smet files: 57,150
   Total:       114,300

⚙️  Loading data...
   This will take 30-60 minutes...
   ✅ Loaded in 45.3 minutes
   Locations: 57,150
   Time points: 730 (2 years of data)

📅 Keeping all historical data (initial mode)

🧭 Processing data...
   Filtering north aspects (315° - 45°)...
   North-facing: 12,458 / 57,150 (21.8%)
   Calculating max HS...

💾 Saving archive to snowpack_operational.nc...
   ✅ Saved in 2.3 minutes
   File size: 3,847.2 MB

======================================================================
✅ INITIAL COMPLETE
======================================================================
Total time: 47.6 minutes

Output:
  Archive: snowpack_operational.nc
  Size: 3,847.2 MB
  Locations: 12,458
  Time points: 730

📝 Next run will use UPDATE mode with 90-day window
======================================================================
```

**Result:** Full historical archive created with 2 years of data!

### Second Run (Update)

```bash
# Same command as before!
$ python snowpack_updater.py /home/snowpack/runs/output.2024

======================================================================
SNOWPACK ARCHIVE UPDATER
======================================================================
Data directory: /home/snowpack/runs/output.2024
Archive file: snowpack_operational.nc

🔍 Mode: UPDATE
   Archive exists - updating with 90-day window
   Last modified: 2024-12-19 15:30:45
======================================================================

📊 Scanning for files...
   .pro files:  57,150
   .smet files: 57,150
   Total:       114,300

⚙️  Loading data...
   ✅ Loaded in 43.2 minutes
   Locations: 57,150
   Time points: 731 (SNOWPACK added 1 timestep)

📅 Applying 90-day time window...
   Cutoff date: 2024-09-20
   Time points before: 731
   Time points after:  90
   Removed: 641 old timesteps

🧭 Processing data...
   North-facing: 12,458 / 57,150 (21.8%)

💾 Saving archive to snowpack_operational.nc...
   ✅ Saved in 1.8 minutes
   File size: 487.3 MB

======================================================================
✅ UPDATE COMPLETE
======================================================================
Total time: 45.0 minutes

Output:
  Archive: snowpack_operational.nc
  Size: 487.3 MB
  Locations: 12,458
  Time points: 90
======================================================================
```

**Result:** Archive updated with rolling 90-day window!

## Command-Line Options

### Required

```bash
data_dir        # Directory with .pro and .smet files
```

### Optional

```bash
--archive FILE           # Archive file path
                        # Default: snowpack_operational.nc

--window-days N         # Time window for updates (days)
                        # Default: 90
                        # Note: Ignored on first run (keeps all history)

--mode {auto,initial,update}
                        # auto: Auto-detect (default)
                        # initial: Force initial load
                        # update: Force update mode

--lazy                  # Use lazy loading (default: True)
--eager                 # Use eager loading
--cpus N               # Number of CPUs (default: 15)
```

## Advanced Usage

### Force Initial Load (Ignore Existing Archive)

```bash
# Useful if you want to regenerate full historical archive
python snowpack_updater.py /data/snowpack \
    --mode initial \
    --archive snowpack_full_history.nc
```

### Force Update Mode (Require Existing Archive)

```bash
# Will error if archive doesn't exist
python snowpack_updater.py /data/snowpack \
    --mode update
```

### Different Time Windows

```bash
# Quick updates (30-day window)
python snowpack_updater.py /data/snowpack \
    --archive snowpack_quick.nc \
    --window-days 30

# Extended operational (180-day window)
python snowpack_updater.py /data/snowpack \
    --archive snowpack_extended.nc \
    --window-days 180
```

## Python API

### Basic Usage

```python
from snowpack_updater import SnowpackUpdater

# Auto-detect mode
updater = SnowpackUpdater(
    data_dir='/home/snowpack/runs/output.2024',
    archive_file='snowpack_operational.nc',
    window_days=90
)

ds = updater.run()
```

### Explicit Mode Control

```python
# Force initial load
updater = SnowpackUpdater(
    data_dir='/data/snowpack',
    archive_file='full_archive.nc',
    mode='initial'  # Force initial, ignore existing
)
ds = updater.run()

# Force update
updater = SnowpackUpdater(
    data_dir='/data/snowpack',
    archive_file='operational.nc',
    mode='update',  # Require existing archive
    window_days=90
)
ds = updater.run()
```

### Accessing Results

```python
updater = SnowpackUpdater(...)
ds = updater.run()

# Check what mode was used
print(f"Mode: {updater.mode}")
print(f"Window: {updater.window_days} days")

# Access data
print(f"Locations: {len(ds.data.location)}")
print(f"Time points: {len(ds.data.time)}")

# Query data
max_hs = ds.data.max_HS.max()
print(f"Maximum HS: {max_hs.values:.2f} m")
```

## Scheduling

### Cron Job (Hourly Updates)

```bash
# /etc/cron.d/snowpack-updates
0 * * * * user cd /home/user && python snowpack_updater.py /data/snowpack >> /var/log/snowpack.log 2>&1
```

### Daily Updates

```bash
# Once per day at 2 AM
0 2 * * * user cd /home/user && python snowpack_updater.py /data/snowpack >> /var/log/snowpack.log 2>&1
```

## Multiple Archives Strategy

### Operational + Historical

```bash
# Daily: Update operational archive (90 days)
python snowpack_updater.py /data/snowpack \
    --archive operational.nc \
    --window-days 90

# Monthly: Regenerate full historical archive
python snowpack_updater.py /data/snowpack \
    --archive historical.nc \
    --mode initial
```

### Multiple Time Windows

```bash
# Quick (30 days) - for rapid updates
python snowpack_updater.py /data/snowpack \
    --archive quick.nc \
    --window-days 30

# Operational (90 days) - main forecasting
python snowpack_updater.py /data/snowpack \
    --archive operational.nc \
    --window-days 90

# Extended (180 days) - seasonal analysis
python snowpack_updater.py /data/snowpack \
    --archive extended.nc \
    --window-days 180
```

## Troubleshooting

### Archive Not Found (Update Mode)

```bash
$ python snowpack_updater.py /data/snowpack --mode update

FileNotFoundError: Update mode specified but archive not found: snowpack_operational.nc
```

**Solution:** Remove `--mode update` to use auto-detect, or create initial archive first.

### Out of Memory

```bash
💾 Memory estimate:
   Available: 32.0 GB
   Required:  40.5 GB
   ⚠️  Memory usage may be high!
```

**Solution:** Use lazy loading (default) or reduce time window:
```bash
python snowpack_updater.py /data/snowpack --lazy --window-days 30
```

### Wrong Mode Detected

```bash
# Archive exists but you want to regenerate
python snowpack_updater.py /data/snowpack --mode initial
```

## Performance Tips

### For Initial Load (First Run)

```bash
# Use lazy loading for large datasets
python snowpack_updater.py /data/snowpack --lazy

# Allocate more CPUs
python snowpack_updater.py /data/snowpack --cpus 20

# Run overnight (takes 45-60 minutes)
nohup python snowpack_updater.py /data/snowpack &
```

### For Updates

```bash
# Reduce window for faster updates
python snowpack_updater.py /data/snowpack --window-days 30

# Use eager if it fits in memory (faster)
python snowpack_updater.py /data/snowpack --eager --window-days 30
```

## Monitoring

### Check Logs

```bash
tail -f snowpack_updates.log
```

### Check Archive

```python
import xsnow

ds = xsnow.read('snowpack_operational.nc')
print(f"Locations: {len(ds.data.location):,}")
print(f"Time points: {len(ds.data.time):,}")
print(f"Date range: {ds.data.time.values[0]} to {ds.data.time.values[-1]}")
```

### Archive Size

```bash
ls -lh snowpack_operational.nc

# Expected sizes:
# - Full history (2 years): ~4 GB
# - 90-day window: ~500 MB
# - 30-day window: ~150 MB
```

## Summary

**Key Points:**
1. ✅ One class handles both initial and update
2. ✅ Auto-detects what to do
3. ✅ Use same command every time
4. ✅ First run: Keeps all history
5. ✅ Updates: Applies rolling window
6. ✅ Simple, reliable, maintainable

**Typical Usage:**
```bash
# Just run this after each SNOWPACK run!
python snowpack_updater.py /home/snowpack/runs/output.2024
```

That's it! 🚀
