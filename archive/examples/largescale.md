# xsnow-largescale Usage Guide

Complete guide to using xsnow-largescale for managing very large SNOWPACK datasets.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [File Change Tracking](#file-change-tracking)
4. [Incremental Updates](#incremental-updates)
5. [Memory Management](#memory-management)
6. [Distributed Processing](#distributed-processing)
7. [Operational Workflows](#operational-workflows)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI (when published)
```bash
pip install xsnow-largescale
```

### From source
```bash
git clone https://github.com/yourusername/xsnow-largescale.git
cd xsnow-largescale
pip install -e .
```

### Verify installation
```bash
python examples/test_extension.py
```

---

## Quick Start

### Basic Usage

```python
import xsnow
import xsnow_largescale  # Auto-registers extension methods

# Load existing archive
ds = xsnow.read('archive_current.nc')

# Update from changed files only (fast!)
ds_updated = ds.update_from_changed_files(
    data_path='/data/pros',
    window_days=30
)

# Save
ds_updated.to_netcdf('archive_current.nc')
```

### What Just Happened?

1. **File tracking**: Extension scanned `/data/pros` and found files that changed
2. **Selective loading**: Only loaded the changed files (typically 1-5% of total)
3. **Intelligent merging**: Updated changed locations, kept unchanged ones
4. **Time windowing**: Kept only last 30 days of data
5. **Fast**: 5 seconds instead of 13 minutes!

---

## File Change Tracking

### How It Works

The extension tracks file modification times in `file_metadata.json`:

```json
{
  "/data/pros/station001.pro": {
    "mtime": 1703001234.567,
    "last_timestamp": "2024-12-17T12:00:00",
    "updated_at": "2024-12-17T14:30:00",
    "size_bytes": 5242880
  },
  ...
}
```

### Manual Tracking

```python
from xsnow_largescale import FileChangeTracker

# Initialize tracker
tracker = FileChangeTracker('metadata.json')

# Find changed files
changed = tracker.find_changed_files('/data/pros')
print(f"Changed: {len(changed)} files")

# After processing
for file in changed:
    tracker.update_metadata(file)
tracker.save_metadata()

# Get statistics
stats = tracker.get_statistics()
print(f"Tracking {stats['total_files']} files")
print(f"Total size: {stats['total_size_mb']:.1f} MB")
```

### Resetting Tracking

```python
# Clear all tracking (forces full reload next time)
tracker.reset(confirm=True)
```

---

## Incremental Updates

### update_from_changed_files()

Most common method - updates by processing only changed files.

```python
ds_updated = ds.update_from_changed_files(
    data_path='/data/pros',           # Directory with .pro files
    metadata_file='metadata.json',     # Tracking file
    window_days=30,                    # Keep last N days
    n_cpus=15,                         # CPU cores to use
    lazy=None                          # Auto-detect
)
```

**Performance:**
- 995 files, 1% changed: ~5 seconds
- 10,000 files, 1% changed: ~30 seconds
- 200,000 files, 1% changed: ~2 minutes (distributed)

### estimate_update_resources()

Check what an update will require before running it.

```python
estimates = ds.estimate_update_resources('/data/pros')

print(f"Changed: {estimates['changed_files']} files")
print(f"Memory: {estimates['estimated_memory_gb']:.1f} GB")
print(f"Time: {estimates['estimated_time_minutes']:.1f} min")
print(f"Strategy: {estimates['strategy']}")
```

**Use this to:**
- Decide if update is feasible
- Choose between eager/lazy/distributed
- Plan maintenance windows

### save_with_tracking()

Convenience method combining save with tracking updates.

```python
ds.save_with_tracking(
    'archive.nc',
    metadata_file='metadata.json'
)
```

---

## Memory Management

### Understanding Memory Usage

```python
from xsnow_largescale import MemoryMonitor

monitor = MemoryMonitor(max_memory_gb=50)

# Check if files will fit
if monitor.will_fit(n_files=2000):
    print("Can load all files at once")
else:
    print("Need batching or lazy loading")

# Get recommendation
batch_size = monitor.recommend_batch_size(n_files=10000)
print(f"Process in batches of {batch_size} files")

# Current usage
usage = monitor.current_usage()
print(f"RAM: {usage['used_gb']:.1f}/{usage['total_gb']:.1f} GB ({usage['percent']:.1f}%)")
```

### update_with_batching()

For memory-constrained systems.

```python
ds_updated = ds.update_with_batching(
    data_path='/data/pros',
    max_memory_gb=32,      # Memory limit
    batch_size=1000,       # Files per batch (or None for auto)
    window_days=30
)
```

**How it works:**
1. Finds changed files
2. Calculates optimal batch size
3. Processes batches sequentially
4. Merges results
5. Applies time window

**Performance tradeoff:**
- Slower than non-batched (merging overhead)
- Enables processing on limited systems
- Still much faster than full reload

---

## Distributed Processing

### Setting Up Dask Cluster

#### Option 1: Local Cluster (Testing)

```python
from xsnow_largescale.distributed import setup_local_cluster

client = setup_local_cluster(
    n_workers=4,
    threads_per_worker=4,
    memory_limit='32GB'
)

print(f"Dashboard: {client.dashboard_link}")

# Use for updates
ds.update_distributed(client=client, ...)

client.close()
```

#### Option 2: Remote Cluster (Production)

**On scheduler server:**
```bash
dask-scheduler
# Scheduler at: tcp://192.168.1.100:8786
```

**On each worker server:**
```bash
dask-worker tcp://192.168.1.100:8786 \
    --nworkers 15 \
    --threads-per-worker 2 \
    --memory-limit 128GB
```

**In Python:**
```python
from dask.distributed import Client

client = Client('tcp://192.168.1.100:8786')
print(f"Workers: {len(client.scheduler_info()['workers'])}")
```

### update_distributed()

For very large datasets (50,000+ files).

```python
from dask.distributed import Client

client = Client('scheduler:8786')

ds_updated = ds.update_distributed(
    data_path='/data/pros',
    client=client,
    window_days=30
)

client.close()
```

**Performance:**
- 200,000 files, 1% changed: ~2 minutes
- Scales linearly with workers
- Requires cluster setup

**When to use:**
- 20,000+ files
- Regular updates
- Available infrastructure

---

## Operational Workflows

### Daily Update (Automated)

```python
#!/usr/bin/env python3
"""Daily update cron job."""

import xsnow
import xsnow_largescale

def main():
    # Load archive
    ds = xsnow.read('archive_current.nc')
    
    # Estimate first
    est = ds.estimate_update_resources('/data/pros')
    
    if est['changed_files'] == 0:
        print("No updates needed")
        return
    
    # Update based on strategy
    if est['strategy'] == 'eager':
        ds = ds.update_from_changed_files('/data/pros')
    else:
        ds = ds.update_with_batching('/data/pros')
    
    # Save
    ds.to_netcdf('archive_current.nc')
    print(f"Updated {est['changed_files']} files")

if __name__ == '__main__':
    main()
```

**Cron setup:**
```bash
# Edit crontab
crontab -e

# Add line (runs at 2 AM daily)
0 2 * * * /path/to/venv/bin/python /path/to/daily_update.py
```

### Weekly Maintenance

```python
"""Weekly archive maintenance."""

def weekly_maintenance():
    # Check tracking consistency
    tracker = FileChangeTracker()
    check = tracker.check_consistency('/data/pros')
    
    print(f"Missing files: {check['n_missing']}")
    print(f"Modified: {check['n_modified']}")
    
    # Compress old data
    ds = xsnow.read('archive_current.nc')
    
    # Keep recent in current, move old to historical
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=7)
    
    recent = ds.sel(time=slice(cutoff.isoformat(), None))
    old = ds.sel(time=slice(None, cutoff.isoformat()))
    
    recent.to_netcdf('archive_current.nc')
    
    # Append to historical
    if Path('archive_historical.nc').exists():
        hist = xsnow.read('archive_historical.nc')
        combined = xsnow.concat([hist, old], dim='time')
        combined.to_netcdf('archive_historical.nc')
    else:
        old.to_netcdf('archive_historical.nc')
```

### Monitoring

```python
"""Monitor update performance."""

from xsnow_largescale import ProgressTracker
import time

tracker = ProgressTracker(total=1000, desc="Processing files")

for i in range(1000):
    # Process file
    time.sleep(0.01)
    tracker.update()

tracker.finish()
# Output: ✅ Processing files complete: 1000 items in 10.2s (98.0 items/sec)
```

---

## Performance Tuning

### Optimal CPU Usage

```python
import os

# Default: use all but one core
n_cpus = os.cpu_count() - 1

# For I/O bound (many small files)
n_cpus = min(32, os.cpu_count())

# For CPU bound (large files)
n_cpus = os.cpu_count() // 2  # Leave room for other processes
```

### Lazy vs Eager Decision

```python
from xsnow_largescale import MemoryMonitor

monitor = MemoryMonitor()
estimated_gb = monitor.estimate_usage(n_files)

if estimated_gb < monitor.max_memory_gb * 0.5:
    lazy = False  # Eager is faster
else:
    lazy = True   # Won't fit in memory
```

### Batch Size Optimization

```python
# Auto (recommended)
batch_size = None  # Let extension decide

# Manual (fine-tuning)
batch_size = 1000  # Process 1000 files at a time

# Based on memory
batch_size = int(available_gb * 200)  # ~200 files per GB
```

---

## Troubleshooting

### Extension Methods Not Available

**Problem:** `AttributeError: 'xsnowDataset' object has no attribute 'update_from_changed_files'`

**Solution:**
```python
# Explicitly register
import xsnow_largescale
xsnow_largescale.register_extensions()

# Verify
methods = xsnow_largescale.list_extensions()
print(f"Available: {methods}")
```

### Memory Errors

**Problem:** `MemoryError` or system becomes unresponsive

**Solution:**
```python
# Check before running
monitor = MemoryMonitor()
estimates = ds.estimate_update_resources('/data/pros')

if estimates['estimated_memory_gb'] > monitor.max_memory_gb:
    # Use batching
    ds = ds.update_with_batching(
        '/data/pros',
        max_memory_gb=available_gb
    )
```

### Slow Updates

**Problem:** Updates taking too long

**Diagnosis:**
```python
# Check what's changed
tracker = FileChangeTracker()
changed = tracker.find_changed_files('/data/pros')
total = len(list(Path('/data/pros').rglob('*.pro')))

print(f"Changed: {len(changed)} / {total} ({len(changed)/total*100:.1f}%)")
```

**Solutions:**
- If many files changed: Normal (initial run or major update)
- If few files changed but slow: Check disk I/O, network if remote
- If always slow: Consider distributed processing

### Tracking Issues

**Problem:** Always processes all files

**Check:**
```python
tracker = FileChangeTracker('metadata.json')
stats = tracker.get_statistics()

if stats['total_files'] == 0:
    print("Tracking file is empty - first run?")
else:
    print(f"Tracking {stats['total_files']} files")
```

**Reset if needed:**
```python
tracker.reset(confirm=True)  # Forces fresh tracking
```

---

## Best Practices

1. **Always estimate first**
   ```python
   estimates = ds.estimate_update_resources('/data/pros')
   # Check before committing to strategy
   ```

2. **Monitor memory usage**
   ```python
   monitor = MemoryMonitor()
   usage = monitor.current_usage()
   if usage['percent'] > 80:
       # System under pressure
   ```

3. **Log everything**
   ```python
   import logging
   logging.basicConfig(
       level=logging.INFO,
       filename='updates.log'
   )
   ```

4. **Test before production**
   ```python
   # Test on small subset first
   test_files = changed_files[:100]
   ds_test = xsnow.read(test_files, ...)
   ```

5. **Keep backups**
   ```python
   # Before update
   import shutil
   shutil.copy('archive.nc', 'archive.nc.backup')
   ```

---

## Support

- Issues: https://github.com/yourusername/xsnow-largescale/issues
- Discussions: https://github.com/yourusername/xsnow-largescale/discussions
- Email: your.email@example.com
