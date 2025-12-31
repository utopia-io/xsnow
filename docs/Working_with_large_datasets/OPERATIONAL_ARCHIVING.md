# Operational SNOWPACK Archiving System

## Overview

A production system for maintaining current-state SNOWPACK archives with concurrent processing, memory optimization, and robust error handling.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Zarr Utilities](#zarr-utilities)
5. [Operational Archive](#operational-archive)
6. [Shell Wrapper](#shell-wrapper)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## Philosophy

### Current State vs Historical Archive

**Traditional Approach (Historical):**
```
Run 1 (6 AM):  Create batch_0600.zarr (14 days of data)
Run 2 (12 PM): Create batch_1200.zarr (14 days of data) 
Run 3 (6 PM):  Create batch_1800.zarr (14 days of data)
Run 4 (12 AM): Create batch_0000.zarr (14 days of data)

Result: 4 files, 90% duplicate data, ~2GB total
```

**Operational Approach (Current State):**
```
Run 1 (6 AM):  Update operational.zarr (14 days of data)
Run 2 (12 PM): Update operational.zarr (latest data, old timestamps replaced)
Run 3 (6 PM):  Update operational.zarr (latest data, old timestamps replaced)
Run 4 (12 AM): Update operational.zarr (latest data, old timestamps replaced)

Result: 1 file, always current, no redundancy, ~500MB
```

### Key Insight

**Operational archive = CURRENT STATE, not history**

For 4x/day operations:
- ✅ One archive file, always up to date
- ✅ New data updates existing timestamps
- ✅ Rolling time window (e.g., last 14 days)
- ✅ No redundant storage
- ✅ Perfect for operational forecasting

### Concurrent Processing

**Problem:** Sequential processing wastes time
```
SNOWPACK run:    60 minutes
Archive update:  45 minutes (sequential)
Total:          105 minutes
```

**Solution:** Concurrent processing with priority
```
SNOWPACK run:     60 minutes (normal priority)
Archive update:   15 minutes (low priority, overlapped)
Total:           ~65 minutes (38% faster)
Time saved:       40 minutes per run
```

**How:**
- Run archive update DURING SNOWPACK execution
- Use `nice` and `ionice` for low priority
- SNOWPACK gets CPU/IO priority
- Archive runs in background
- Both complete faster

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Shell Wrapper Layer                     │
│  run_operational_update_zarr.sh                         │
│  - Safety checks (disk, memory, locks)                  │
│  - Process priority (nice/ionice)                       │
│  - Archive naming (rolling/timestamped)                 │
│  - Logging and error handling                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Python Archive Layer                        │
│  operational_archive_zarr.py                            │
│  - File change detection                                │
│  - Batch processing (memory-efficient)                  │
│  - Merge with existing archive                          │
│  - Time window management                               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Shared Utilities Layer                      │
│  zarr_utils.py                                          │
│  - Zarr encoding (fill values)                          │
│  - Adaptive chunking                                    │
│  - Coordinate preprocessing                             │
│  - File operations                                      │
│  - Batch management                                     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
SNOWPACK Files (.pro/.smet)
         ↓
   File Scanner (tracks changes)
         ↓
   Batch Creator (2000 files/batch)
         ↓
   xsnow.read() [lazy loading]
         ↓
   Preprocessing (coordinate handling)
         ↓
   Temp Zarr Batches
         ↓
   Merge Operation (coords='minimal')
         ↓
   Combine with Old Archive
         ↓
   Deduplicate Times
         ↓
   Apply Time Window
         ↓
   Final Archive (operational.zarr)
```

---

## Components

### 1. zarr_utils.py - Shared Utilities

**Purpose:** Reusable functions for Zarr operations, eliminating code duplication.

**Functions:**

#### File Operations
```python
find_snowpack_files(dir, pattern='both')
# Recursively find .pro and/or .smet files
# pattern: '*.pro', '*.smet', or 'both'

filter_files_by_mtime(files, min_age_seconds=0, max_age_seconds=None)
# Filter files by modification time
# Useful for skipping recently modified files

create_file_batches(files, batch_size=2000)
# Split file list into batches for memory management
```

#### Zarr Encoding
```python
get_zarr_encoding(ds)
# Returns encoding dict with dtype-specific fill values
# int8: -128, int16/32/64: -999 (SNOWPACK standard)

FILL_VALUES = {
    'int8': -128,      # Fits in range
    'int16': -999,     # SNOWPACK standard
    'int32': -999,
    'int64': -999,
    'uint8': 255,
    'uint16': 65535,
    ...
}
```

#### Chunking
```python
get_optimal_chunks(ds)
# Adaptive chunking based on available dimensions
# Only chunks dimensions that exist (handles .pro and .smet)
# location: 200, time: 100, layer/slope: -1 (no chunking)
```

#### Preprocessing
```python
preprocess_for_merge(ds)
# 1. Convert slope coordinates (azimuth, inclination) to data variables
#    - Handles 1D → 2D broadcasting
# 2. Drop derived coordinates (z, x, y)
# 3. Keep essential coordinates (lat, lon, altitude)
```

#### Merging
```python
merge_zarr_batches(batch_files, concat_dim='location', preprocess=True)
# Load multiple Zarr batches and merge
# Uses coords='minimal' to handle dimension mismatches

remove_duplicate_times(ds)
# Use numpy.unique to find and remove duplicate timesteps
# Sorts indices to maintain temporal order
```

#### Saving
```python
save_to_zarr(ds, output_path, mode='w')
# Save with proper encoding
# Applies fill values for integer types
# Zarr v3 compression applied automatically
```

**Why These Functions?**
- Single source of truth for Zarr operations
- Fix bugs once, fixed everywhere
- Testable independently
- Easy to extend

---

### 2. operational_archive_zarr.py - Python Core

**Purpose:** Detect changed files, process in batches, merge with archive.

#### Class: OperationalArchive

**Initialization:**
```python
OperationalArchive(
    data_dir='/data/snowpack',           # SNOWPACK output directory
    archive_file='operational.zarr',     # Output archive
    tracking_file='file_tracking.json',  # Change tracking
    window_days=14,                      # Rolling time window
    batch_size=2000,                     # Files per batch (memory control)
    n_cpus=8,                            # Parallel processing
    modification_threshold=60            # Skip files modified <60s ago
)
```

**Key Methods:**

##### update()
Main entry point for archive update.
```python
def update(self):
    # 1. Load file tracking (what changed since last run)
    # 2. Find changed files (modified since last run)
    # 3. Create temp batches (process in chunks)
    # 4. Merge with archive (combine old + new)
    # 5. Apply time window (keep last N days)
    # 6. Update tracking (record new modification times)
    # 7. Cleanup (remove temp files)
```

##### _load_tracking()
Track file modification times to detect changes.
```python
def _load_tracking(self):
    # JSON format: {relative_path: mtime}
    # Initializes with all files on first run
    # Uses zarr_utils.find_snowpack_files()
```

##### _find_changed_files()
Identify files that changed since last run.
```python
def _find_changed_files(self):
    # 1. Find all SNOWPACK files
    # 2. Filter out very recent files (still being written)
    # 3. Compare mtime with tracked mtime
    # 4. Return list of changed files
```

##### _create_temp_batches()
Process files in memory-efficient batches.
```python
def _create_temp_batches(self, changed_files):
    # 1. Split into batches (batch_size files each)
    # 2. For each batch:
    #    - Read with xsnow.read(lazy=True)  # Memory efficient!
    #    - Preprocess for merge
    #    - Rechunk for Zarr
    #    - Save to temp batch file
    # 3. Return list of temp batch files
```

**Critical:** Uses `lazy=True` to avoid loading entire batch into memory!

##### _merge_with_archive()
Combine new data with existing archive.
```python
def _merge_with_archive(self, new_batches):
    # 1. Merge new batches (location dimension)
    #    Uses zarr_utils.merge_zarr_batches()
    # 2. Load existing archive if present
    # 3. Concat along time dimension
    #    Uses coords='minimal' to handle .pro/.smet differences
    # 4. Remove duplicate times
    #    Uses zarr_utils.remove_duplicate_times()
    # 5. Rechunk and save
```

**Key:** `coords='minimal'` allows merging .pro files (with layer) and .smet files (without layer).

##### _apply_time_window()
Keep only recent data within rolling window.
```python
def _apply_time_window(self):
    # 1. Calculate cutoff date (now - window_days)
    # 2. Convert to numpy.datetime64 (critical for comparison!)
    # 3. Filter times >= cutoff
    # 4. Save windowed dataset
```

**Fix:** Uses `np.datetime64(cutoff)` not `cutoff.timestamp()` to avoid dtype mismatch.

##### _update_tracking()
Record new modification times for next run.
```python
def _update_tracking(self, changed_files):
    # Update file_tracking dict with new mtimes
    # Save to JSON
```

#### Command-Line Interface

```bash
python operational_archive_zarr.py /data/snowpack \
    --archive operational.zarr \
    --window-days 14 \
    --batch-size 2000 \
    --cpus 8 \
    --modification-threshold 60
```

---

### 3. run_operational_update_zarr.sh - Shell Wrapper

**Purpose:** Production wrapper with safety checks, process priority, and error handling.

#### Configuration

```bash
# Archive settings
WINDOW_DAYS=14
ARCHIVE_NAMING="timestamped"      # or "rolling"
ARCHIVE_BASENAME="operational"

# Process priority (concurrent execution)
NICE_LEVEL=19                     # CPU priority (19=lowest)
IONICE_CLASS=2                    # I/O scheduling class
IONICE_LEVEL=7                    # I/O priority (7=lowest)

# Resource limits
MIN_DISK_GB=50                    # Minimum free disk space
MIN_MEMORY_GB=40                  # Minimum free memory
MAX_RUNTIME_HOURS=3               # Kill if exceeds

# Paths
PYTHON_SCRIPT="${SCRIPT_DIR}/operational_archive_zarr.py"
LOG_DIR="./logs/snowpack-archive"
LOCK_FILE="./snowpack-archive.lock"
```

#### Key Functions

##### get_archive_filename()
Generate archive filename based on naming mode.

**Rolling Mode:**
```bash
ARCHIVE_NAMING="rolling"
# Result: operational.zarr (same file updated each run)
```

**Timestamped Mode:**
```bash
ARCHIVE_NAMING="timestamped"
# Detects SNOWPACK runtime from newest .pro file
# Result: operational_2025-12-15_09:53_2025-12-29_09:53.zarr
```

**How it works:**
```bash
# Find newest .pro file
newest_file=$(find "${data_dir}" -name "*.pro" -type f -printf '%T@ %p\n' | sort -rn | head -1)

# Get modification time (SNOWPACK end time)
end_timestamp=$(stat -c %Y "${newest_file}")
end_date=$(date -d "@${end_timestamp}" +"%Y-%m-%d_%H:%M")

# Calculate start time (end - window_days)
start_timestamp=$((end_timestamp - WINDOW_DAYS * 86400))
start_date=$(date -d "@${start_timestamp}" +"%Y-%m-%d_%H:%M")

# Generate filename
archive="./data/${ARCHIVE_BASENAME}_${start_date}_${end_date}.zarr"
```

**Critical:** Function outputs ONLY the filename to stdout. No log() calls inside function to avoid polluting the output.

##### get_priority_command()
Build process priority command.
```bash
get_priority_command() {
    local cmd="nice -n ${NICE_LEVEL}"
    
    # Add ionice if available
    if command -v ionice >/dev/null 2>&1; then
        cmd="${cmd} ionice -c ${IONICE_CLASS} -n ${IONICE_LEVEL}"
    fi
    
    echo "${cmd}"
}

# Usage:
PRIORITY_CMD=$(get_priority_command)
# Result: "nice -n 19 ionice -c 2 -n 7"
```

##### check_disk_space()
Verify sufficient disk space available.
```bash
check_disk_space() {
    local mount_point="$1"
    local free_gb=$(df -BG "${mount_point}" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "${free_gb}" -lt "${MIN_DISK_GB}" ]; then
        log_error "Insufficient disk space: ${free_gb}GB free, need ${MIN_DISK_GB}GB"
        return 1
    fi
    
    log "Disk space OK: ${free_gb}GB free"
    return 0
}
```

##### check_snowpack_running()
Check if SNOWPACK is currently running.
```bash
check_snowpack_running() {
    if pgrep -f "snowpack" > /dev/null; then
        log "SNOWPACK is running - will process files concurrently"
        return 0
    else
        log "SNOWPACK not running - processing finalized files"
        return 0
    fi
}
```

Note: This is informational only. Archive update runs regardless (concurrent or after completion).

##### acquire_lock() / release_lock()
Prevent multiple simultaneous updates.
```bash
acquire_lock() {
    if [ -e "${LOCK_FILE}" ]; then
        local lock_pid=$(cat "${LOCK_FILE}")
        
        # Check if process still running
        if kill -0 "${lock_pid}" 2>/dev/null; then
            log_error "Another update is running (PID: ${lock_pid})"
            return 1
        else
            log "Removing stale lock file"
            rm -f "${LOCK_FILE}"
        fi
    fi
    
    echo $$ > "${LOCK_FILE}"
    log "Acquired lock"
    return 0
}
```

##### cleanup()
Exit handler for graceful shutdown.
```bash
cleanup() {
    local exit_code=$?
    
    log "Cleaning up (exit code: ${exit_code})"
    
    # Kill update if still running
    if [ -n "${UPDATE_PID:-}" ]; then
        if kill -0 "${UPDATE_PID}" 2>/dev/null; then
            log "Killing update process ${UPDATE_PID}"
            kill -TERM "${UPDATE_PID}" 2>/dev/null || true
            sleep 2
            kill -KILL "${UPDATE_PID}" 2>/dev/null || true
        fi
    fi
    
    release_lock
    
    if [ ${exit_code} -ne 0 ]; then
        send_notification "Archive Update FAILED" "Exit code: ${exit_code}"
    fi
    
    exit ${exit_code}
}

# Register cleanup
trap cleanup EXIT INT TERM
```

#### Main Execution Flow

```bash
# 1. Parse arguments
DATA_DIR="$1"

# 2. Create log directory
mkdir -p "${LOG_DIR}"

# 3. Determine archive filename
ARCHIVE_FILE=$(get_archive_filename "${DATA_DIR}" 2>/dev/null)

# 4. Pre-flight checks
check_disk_space "$(dirname "${ARCHIVE_FILE}")"
check_memory
check_snowpack_running
check_recent_snowpack_run "${DATA_DIR}"
acquire_lock

# 5. Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    log_error "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# 6. Execute with priority and timeout
PRIORITY_CMD=$(get_priority_command)
${PRIORITY_CMD} timeout ${MAX_RUNTIME_HOURS}h python3 "${PYTHON_SCRIPT}" \
    "${DATA_DIR}" \
    --archive "${ARCHIVE_FILE}" \
    --window-days ${WINDOW_DAYS} \
    --batch-size 2000 \
    --cpus 8 \
    >> "${LOG_DIR}/operational.log" 2>&1 &

UPDATE_PID=$!
log "Update started (PID: ${UPDATE_PID})"

# 7. Wait for completion
wait ${UPDATE_PID}
EXIT_CODE=$?

# 8. Report results
if [ ${EXIT_CODE} -eq 0 ]; then
    log "✅ UPDATE COMPLETE"
else
    log_error "❌ UPDATE FAILED (exit code: ${EXIT_CODE})"
fi
```

---

## Configuration

### Memory Management

**Batch Size:**
- Small (1000): Safest, slowest
- Medium (2000): Recommended balance
- Large (5000): Faster but needs more RAM

**Rule of thumb:** `batch_size × 5MB ≈ RAM per batch`

**Examples:**
- 2000 files × 5MB = ~10GB RAM ✅ Safe
- 5000 files × 5MB = ~25GB RAM ⚠️ Monitor
- 10000 files × 5MB = ~50GB RAM ❌ Too much

### Process Priority

**For concurrent execution:**
```bash
NICE_LEVEL=19      # Lowest CPU priority
IONICE_CLASS=2     # Best-effort I/O
IONICE_LEVEL=7     # Lowest I/O priority
```

**SNOWPACK always gets priority!**

### Archive Naming

**Rolling (Operational):**
```bash
ARCHIVE_NAMING="rolling"
# Pro: Always same filename, easy to reference
# Con: No historical versions
# Use: Operational forecasting, always want latest
```

**Timestamped (Archival):**
```bash
ARCHIVE_NAMING="timestamped"
# Pro: Creates dated archives, preserves history
# Con: Files accumulate, needs management
# Use: Long-term storage, need historical record
```

**Hybrid Approach:**
```bash
# Script 1: Rolling mode, runs 4x/day (operations)
ARCHIVE_NAMING="rolling"
ARCHIVE_FILE="operational.zarr"

# Script 2: Timestamped mode, runs 1x/week (archival)
ARCHIVE_NAMING="timestamped"
ARCHIVE_FILE="archive_weekly.zarr"
```

---

## Deployment

### Directory Structure

```
/project/
├── operational_archive_zarr.py      # Python core
├── run_operational_update_zarr.sh   # Shell wrapper
├── zarr_utils.py                    # Shared utilities
├── data/
│   └── operational.zarr             # Archive (created by script)
├── logs/
│   └── snowpack-archive/
│       ├── operational.log          # Python output
│       └── wrapper.log              # Shell wrapper output
└── temp_operational/                # Temp batches (auto-cleanup)
    ├── batch_0001.zarr
    └── batch_0002.zarr
```

### Installation

```bash
# 1. Copy files
cp zarr_utils.py /project/
cp operational_archive_zarr.py /project/
cp run_operational_update_zarr.sh /project/
chmod +x run_operational_update_zarr.sh

# 2. Edit configuration in shell script
vim run_operational_update_zarr.sh
# Set: WINDOW_DAYS, ARCHIVE_NAMING, NICE_LEVEL, etc.

# 3. Test run
cd /project
./run_operational_update_zarr.sh /data/snowpack/output

# 4. Add to cron (4x daily)
crontab -e
0 6,12,18,0 * * * cd /project && ./run_operational_update_zarr.sh /data/snowpack/output
```

### Monitoring

**Check status:**
```bash
# Watch progress
tail -f logs/snowpack-archive/operational.log

# Check if running
ps aux | grep operational_archive

# Check memory
watch -n 5 'free -h'

# Check archive size
du -sh data/*.zarr
```

**Verify archive:**
```bash
python << 'EOF'
import xarray as xr
ds = xr.open_zarr('data/operational.zarr')
print(f"Locations: {len(ds.location):,}")
print(f"Times: {len(ds.time):,}")
print(f"Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
ds.close()
EOF
```

---

## Troubleshooting

### High Memory Usage

**Symptom:** RAM usage >50GB

**Causes:**
- Batch size too large
- Eager loading (`lazy=False`)

**Fix:**
```bash
# Reduce batch size
python operational_archive_zarr.py /data \
    --batch-size 1000 \  # Reduce from 2000
    --cpus 4             # Reduce workers
```

### Coordinate Mismatch Errors

**Symptom:** `'layer' not present in all datasets`

**Cause:** Mixing .pro files (have layer) with .smet files (no layer)

**Fix:** Already handled by `coords='minimal'` in concat operations.

**Verify fix is present:**
```bash
grep "coords='minimal'" operational_archive_zarr.py
# Should find it in merge operations
```

### Datetime Comparison Errors

**Symptom:** `DTypePromotionError` or `ufunc 'greater_equal' did not contain a loop`

**Cause:** Comparing datetime64 with float

**Fix:** Already handled by using `np.datetime64(cutoff)` in `_apply_time_window()`

**Verify fix:**
```bash
grep "np.datetime64" operational_archive_zarr.py
# Should find it in time window filtering
```

### Script Exits Silently

**Symptom:** No output, no errors

**Cause:** Missing xarray/numpy (not in correct environment)

**Fix:**
```bash
# Activate xsnow venv
cd /path/to/xsnow
source .venv/bin/activate

# Verify
which python
python -c "import xarray; print('OK')"

# Then run script
./run_operational_update_zarr.sh /data/snowpack
```

### Lock File Issues

**Symptom:** "Another update is running"

**Cause:** Stale lock file from crashed run

**Fix:**
```bash
# Check if actually running
ps aux | grep operational_archive

# If NOT running, remove lock
rm -f snowpack-archive.lock

# If running, wait for completion or kill:
pkill -f operational_archive_zarr.py
```

### Files Not Updating

**Symptom:** Archive not changing despite new SNOWPACK data

**Cause:** Modification threshold filtering out recent files

**Fix:**
```bash
# Reduce threshold (or remove it)
python operational_archive_zarr.py /data \
    --modification-threshold 0  # Don't skip recent files
```

---

## Performance Tuning

### For Different Dataset Sizes

**Small (<10k files):**
```bash
--batch-size 5000
--cpus 15
# Fast processing, higher memory
```

**Medium (10k-50k files):**
```bash
--batch-size 2000  # Recommended
--cpus 8
# Balanced
```

**Large (>50k files):**
```bash
--batch-size 1000
--cpus 4
# Conservative, stable
```

### Time Optimization

**Factors affecting runtime:**
1. Number of changed files (only processes changes!)
2. Batch size (larger = fewer batches = faster)
3. CPU count (parallel processing)
4. Lazy loading (critical for memory)
5. Concurrent execution (overlap with SNOWPACK)

**Typical performance (114k files):**
- Sequential (old): 105 minutes
- Concurrent (new): 65 minutes
- **Savings: 40 minutes per run**

**Yearly savings (4x/day):**
- 40 min × 4 runs × 365 days = 58,400 minutes
- **912 hours = 38 days saved per year!**

---

## Best Practices

### DO:
- ✅ Use rolling mode for operations (always current)
- ✅ Use timestamped mode for archival (preserve history)
- ✅ Run concurrently with SNOWPACK (time savings)
- ✅ Monitor memory usage first few runs
- ✅ Keep batch size ≤2000 for stability
- ✅ Use file tracking (only process changes)
- ✅ Set modification threshold to skip active files

### DON'T:
- ❌ Use eager loading (`lazy=True` is critical!)
- ❌ Set batch size >5000 (memory risk)
- ❌ Run multiple updates simultaneously (lock prevents this)
- ❌ Edit archive file directly (use script only)
- ❌ Remove temp files manually (auto-cleanup)
- ❌ Skip the xsnow venv (imports will fail)

---

## Summary

**Key Features:**
- ✅ Current-state approach (no redundancy)
- ✅ Concurrent processing (38% time savings)
- ✅ Memory optimized (lazy loading, small batches)
- ✅ Robust error handling (locks, timeouts, cleanup)
- ✅ Flexible naming (rolling/timestamped)
- ✅ Production ready (monitoring, logging, recovery)

**Architecture:**
- Python core (operational_archive_zarr.py)
- Shell wrapper (safety + priority)
- Shared utilities (zarr_utils.py)

**Result:**
- One archive file, always current
- Minimal storage, maximum efficiency
- Runs during SNOWPACK execution
- Saves 40 minutes per run

**Perfect for operational forecasting workflows running multiple times per day.**
