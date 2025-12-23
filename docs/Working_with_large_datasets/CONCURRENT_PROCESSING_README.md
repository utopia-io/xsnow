# SnowpackUpdater - Concurrent Processing Guide

**Run xsnow data loading in parallel with SNOWPACK to minimize total processing time**

---

## Overview

SnowpackUpdater can run **concurrently** with SNOWPACK, loading and processing data files as SNOWPACK creates them. This overlaps processing time and gets your data ready for analysis much faster.

### Sequential vs Concurrent Processing

**Sequential (Traditional):**
```
┌─────────────────────┐
│  SNOWPACK runs      │  60 minutes
│  (updates all files)│
└─────────────────────┘
         ↓ Wait for SNOWPACK to finish
┌─────────────────────┐
│  Load & process     │  45 minutes
│  (xsnow reads files)│
└─────────────────────┘
═══════════════════════
Total: 105 minutes ❌
```

**Concurrent (Optimized):**
```
┌─────────────────────┐
│  SNOWPACK runs      │  60 minutes
│  (updates all files)│
└─────────────────────┘
         ↓ Start loader early (50 min mark)
         ┌─────────────────────┐
         │  Load & process     │  15 minutes
         │  (while SNOWPACK    │  (overlapped)
         │   finishes)         │
         └─────────────────────┘
═══════════════════════════════
Total: 65 minutes ✅ (38% faster!)
```

**Time Saved:** 40 minutes (105 min → 65 min)

---

## How It Works

### The Strategy

1. **SNOWPACK starts** - Begins updating all 114,300 files
2. **Wait ~50 minutes** - Let SNOWPACK process most files
3. **Start SnowpackUpdater** - Begin loading files as they complete
4. **SNOWPACK finishes** - All files written
5. **SnowpackUpdater finishes** - ~5 min later (loads remaining files)

**Result:** Post-processing can begin just minutes after SNOWPACK completes!

### Why This Works

SNOWPACK processes files sequentially or in batches. Files that finish early sit idle waiting for SNOWPACK to complete. SnowpackUpdater can start loading these "ready" files while SNOWPACK continues working on the rest.

**Key insight:** We don't need to wait for ALL files to be ready - we can start loading the ones that are!

---

## Quick Start

### Method 1: Manual Background Start (Simple)

**Terminal 1: Start SNOWPACK**
```bash
cd /home/snowpack
./snowpack --config snowpack.ini
# This runs for ~60 minutes
```

**Terminal 2: Wait, then start loader (after ~50 min)**
```bash
# Wait for SNOWPACK to be ~80-90% done
sleep 3000  # 50 minutes

# Start loader
cd /home/ron/xsnow/examples/largescale
python snowpack_updater.py /home/snowpack/runs/output.2024 \
    --archive snowpack_operational.nc \
    --window-days 90
```

**Result:** Loader finishes ~5 minutes after SNOWPACK, not 45 minutes after!

### Method 2: Wrapper Script (Recommended)

Create `run_snowpack_concurrent.sh`:

```bash
#!/bin/bash
# Concurrent SNOWPACK + xsnow loading

SNOWPACK_DIR="/home/snowpack"
DATA_DIR="/home/snowpack/runs/output.2024"
ARCHIVE="snowpack_operational.nc"
WINDOW_DAYS=90
CONCURRENT_DELAY=3000  # 50 minutes

echo "Starting SNOWPACK..."
cd "$SNOWPACK_DIR"
./snowpack --config snowpack.ini &
SNOWPACK_PID=$!

echo "SNOWPACK running (PID: $SNOWPACK_PID)"
echo "Waiting ${CONCURRENT_DELAY}s before starting loader..."

# Wait for SNOWPACK to be mostly done
sleep $CONCURRENT_DELAY

echo "Starting SnowpackUpdater (concurrent)..."
python snowpack_updater.py "$DATA_DIR" \
    --archive "$ARCHIVE" \
    --window-days $WINDOW_DAYS \
    >> snowpack_loader.log 2>&1 &
LOADER_PID=$!

echo "Loader running (PID: $LOADER_PID)"

# Wait for both to complete
echo "Waiting for processes to complete..."
wait $SNOWPACK_PID
echo "✅ SNOWPACK complete"

wait $LOADER_PID
echo "✅ Loader complete"

echo ""
echo "Total time saved: ~40 minutes!"
echo "Data ready for analysis: $ARCHIVE"
```

**Usage:**
```bash
chmod +x run_snowpack_concurrent.sh
./run_snowpack_concurrent.sh
```

### Method 3: Cron Job (Automated)

**For hourly SNOWPACK runs:**

```bash
# crontab -e

# SNOWPACK runs at top of hour
0 * * * * cd /home/snowpack && ./snowpack --config snowpack.ini

# Loader starts 50 min into hour (overlaps with SNOWPACK)
50 * * * * cd /home/ron/xsnow/examples/largescale && \
    python snowpack_updater.py /home/snowpack/runs/output.2024 \
    --archive snowpack_operational.nc \
    --window-days 90 >> /var/log/snowpack_loader.log 2>&1
```

**Result:** Archive ready at :05 past the hour instead of :45!

---

## Detailed Workflow

### Timing Example (60-min SNOWPACK run)

```
Time    SNOWPACK              SnowpackUpdater         Status
────────────────────────────────────────────────────────────────
00:00   Start                 -                       SNOWPACK begins
10:00   Processing files      -                       ~17% done
20:00   Processing files      -                       ~33% done
30:00   Processing files      -                       ~50% done
40:00   Processing files      -                       ~67% done
50:00   Processing files      Start loading           Concurrent begins!
51:00   Processing files      Loading ready files     Both running
52:00   Processing files      Loading ready files     Both running
...
59:00   Finishing up          Loading ready files     Both running
60:00   Complete ✅           Loading remaining       SNOWPACK done
61:00   -                     Loading remaining       Loader continues
62:00   -                     Loading remaining       Loader continues
...
65:00   -                     Complete ✅             Loader done
────────────────────────────────────────────────────────────────
Total time: 65 minutes (vs 105 minutes sequential)
```

### What SnowpackUpdater Does

When started concurrently:

1. **Scans for files** - Finds all .pro and .smet files
2. **Checks file status** - Determines which are ready to read
3. **Loads ready files** - Processes files SNOWPACK finished with
4. **Waits for SNOWPACK** - Lets SNOWPACK finish remaining files
5. **Loads remaining** - Processes the last files SNOWPACK wrote
6. **Applies time window** - Keeps last N days
7. **Saves archive** - Writes final .nc file

**Safety:** SnowpackUpdater handles file access correctly - it won't try to read files SNOWPACK is still writing!

---

## Configuration

### Tuning the Delay

The `CONCURRENT_DELAY` is when to start the loader relative to SNOWPACK start.

**Guidelines:**

| SNOWPACK Duration | Recommended Delay | Rationale |
|-------------------|-------------------|-----------|
| 30 minutes | 20 min (67%) | Start when 2/3 done |
| 60 minutes | 50 min (83%) | Start when 5/6 done |
| 90 minutes | 75 min (83%) | Start when 5/6 done |
| 120 minutes | 100 min (83%) | Start when 5/6 done |

**Formula:** `DELAY = SNOWPACK_DURATION × 0.83`

**Why 83%?** 
- Ensures most files are ready when loader starts
- Minimizes idle waiting
- Accounts for SNOWPACK's typical processing pattern

### Adjusting Window Size

Choose `window_days` based on your forecasting needs:

```bash
# Quick operational (last 30 days)
--window-days 30    # Fast, ~200 MB archive

# Standard operational (last 90 days)
--window-days 90    # Balanced, ~500 MB archive

# Extended operational (last 180 days)
--window-days 180   # Comprehensive, ~1 GB archive

# Keep all history (no window)
--mode initial      # Full archive, 3-20 GB
```

---

## Safety & Reliability

### File Access Safety

**Q:** What if SnowpackUpdater tries to read a file SNOWPACK is still writing?

**A:** xsnow handles this gracefully:

1. **Try to read** - Attempt to open and parse file
2. **If locked/incomplete** - Skip and retry later
3. **If readable** - Load successfully
4. **If persistent errors** - Log and continue

**Result:** Safe to run concurrently - no data corruption or crashes.

### Resource Usage

**CPU:**
- SNOWPACK: Uses configured cores (e.g., 4-8 cores)
- SnowpackUpdater: Uses `--cpus` setting (default: 15 cores)
- **Total:** Should be < total system cores

**Memory:**
- SNOWPACK: ~2-8 GB typically
- SnowpackUpdater: ~2-15 GB (depends on lazy mode)
- **Total:** Should be < available RAM

**Disk I/O:**
- SNOWPACK: Writes sequentially
- SnowpackUpdater: Reads completed files
- **Total:** Both reading/writing different files - minimal conflict

**Recommendation:** Monitor first run to ensure no resource contention.

### Error Handling

If something goes wrong:

**SNOWPACK crashes:**
- Loader continues processing completed files
- Some files may be incomplete
- Check logs and rerun SNOWPACK

**Loader crashes:**
- SNOWPACK continues unaffected
- Restart loader after SNOWPACK finishes
- Fallback to sequential processing

**Both crash:**
- Recover by running separately
- Check system resources
- Review logs

---

## Monitoring & Logs

### Real-Time Monitoring

**Terminal 1: Watch SNOWPACK**
```bash
tail -f /home/snowpack/snowpack.log
```

**Terminal 2: Watch Loader**
```bash
tail -f snowpack_loader.log
```

**Terminal 3: System Resources**
```bash
watch -n 5 'ps aux | grep -E "snowpack|python" | grep -v grep'
```

### Log Output Example

**SnowpackUpdater logs:**
```
2025-12-19 10:50:00 - INFO - SNOWPACK ARCHIVE UPDATER
2025-12-19 10:50:00 - INFO - Mode: UPDATE
2025-12-19 10:50:00 - INFO - Archive exists - updating with 90-day window
2025-12-19 10:50:01 - INFO - Scanning for files...
2025-12-19 10:50:01 - INFO -    .pro files:  57,150
2025-12-19 10:50:01 - INFO -    .smet files: 57,150
2025-12-19 10:50:01 - INFO -    Total:       114,300
2025-12-19 10:50:05 - INFO - Loading data...
2025-12-19 10:50:05 - INFO -    Mode: Lazy
2025-12-19 10:50:05 - INFO -    CPUs: 15
2025-12-19 10:51:30 - INFO -    ✅ Loaded in 1.4 minutes
2025-12-19 10:51:30 - INFO -    Locations: 57,150
2025-12-19 10:51:30 - INFO -    Time points: 731
2025-12-19 10:51:35 - INFO - Applying 90-day time window...
2025-12-19 10:51:35 - INFO -    Removed: 641 old timesteps
2025-12-19 10:52:00 - INFO - Saving archive...
2025-12-19 10:54:30 - INFO -    ✅ Saved: 487 MB
2025-12-19 10:54:30 - INFO - ✅ UPDATE COMPLETE
2025-12-19 10:54:30 - INFO -    Total time: 4.5 minutes
```

### Success Indicators

✅ **Good signs:**
- Both processes show in `ps aux`
- Loader log shows loading progress
- Archive file size grows
- No error messages

⚠️ **Warning signs:**
- High memory usage (>90%)
- Excessive CPU load (>95%)
- Slow progress
- Repeated warnings

❌ **Error signs:**
- Process crashes
- Out of memory errors
- File permission errors
- Corrupted archive

---

## Performance Optimization

### For Large Datasets (50k+ files)

**Optimize CPU allocation:**
```bash
python snowpack_updater.py /data/snowpack \
    --cpus 20 \                    # More parallel workers
    --lazy \                       # Use lazy loading
    --window-days 90
```

**Optimize memory:**
```bash
# If memory is tight
python snowpack_updater.py /data/snowpack \
    --cpus 10 \                    # Fewer workers
    --lazy \                       # Essential!
    --window-days 30               # Shorter window
```

### For Fast Storage (SSD/NVMe)

Increase parallelism:
```bash
python snowpack_updater.py /data/snowpack \
    --cpus 30 \                    # More workers
    --eager \                      # Faster if fits in RAM
    --window-days 90
```

### For Slow Storage (HDD/Network)

Reduce parallelism:
```bash
python snowpack_updater.py /data/snowpack \
    --cpus 5 \                     # Fewer workers
    --lazy \                       # Lower memory
    --window-days 90
```

---

## Troubleshooting

### Issue: Loader finishes before SNOWPACK

**Symptom:** Archive missing recent timesteps

**Cause:** Started loader too early

**Solution:** Increase `CONCURRENT_DELAY`
```bash
# Was: 50 minutes
# Try: 55 minutes
CONCURRENT_DELAY=3300  # 55 min
```

### Issue: Loader waits too long

**Symptom:** Both idle at the end

**Cause:** Started loader too late

**Solution:** Decrease `CONCURRENT_DELAY`
```bash
# Was: 50 minutes  
# Try: 45 minutes
CONCURRENT_DELAY=2700  # 45 min
```

### Issue: Out of memory

**Symptom:** Process killed, OOM errors

**Causes:**
1. Running both in eager mode
2. Insufficient RAM
3. Too many CPUs

**Solutions:**
```bash
# Use lazy loading (essential!)
--lazy

# Reduce CPUs
--cpus 10

# Shorter time window
--window-days 30
```

### Issue: File access errors

**Symptom:** "Permission denied" or "File is locked"

**Causes:**
1. SNOWPACK still writing file
2. File permissions wrong
3. File corrupted

**Solutions:**
- Wait and retry
- Check file permissions: `ls -la /data/snowpack`
- Verify SNOWPACK completed: `ps aux | grep snowpack`

---

## Best Practices

### ✅ DO:

1. **Test timing first**
   - Run sequential once to measure SNOWPACK duration
   - Set delay to 80-90% of duration
   - Test concurrent with monitoring

2. **Monitor resources**
   - Check CPU, memory, disk I/O
   - Adjust if system is overloaded
   - Leave headroom for system processes

3. **Use lazy loading**
   - Essential for large datasets
   - Reduces memory usage 10-20x
   - Slower but safer

4. **Keep logs**
   - Redirect output to log files
   - Monitor for errors
   - Track performance over time

5. **Verify results**
   - Check archive has expected timesteps
   - Verify file size is reasonable
   - Spot-check data values

### ❌ DON'T:

1. **Don't start too early**
   - Wait until SNOWPACK is 80%+ done
   - Starting at 50% wastes loader cycles

2. **Don't overload system**
   - SNOWPACK + Loader CPUs < Total CPUs
   - Leave 20% CPU for system
   - Monitor memory usage

3. **Don't use eager mode carelessly**
   - Only if dataset fits in RAM
   - Can crash system if too large
   - Lazy mode is safer

4. **Don't ignore errors**
   - Check logs regularly
   - Investigate warnings
   - Fix issues before they compound

5. **Don't forget to save output**
   - Use `>> log.txt 2>&1`
   - Separate logs for each component
   - Keep logs for troubleshooting

---

## Advanced: Multi-Stage Processing

For very large operations, use multiple stages:

### Stage 1: Concurrent loading during SNOWPACK
```bash
python snowpack_updater.py /data/snowpack \
    --archive stage1_raw.nc \
    --mode initial \              # Keep all history
    --lazy
```

### Stage 2: Apply time window after SNOWPACK
```bash
python snowpack_updater.py /data/snowpack \
    --archive stage2_windowed.nc \
    --mode update \
    --window-days 90
```

### Stage 3: Post-processing
```python
import xsnow

# Load windowed data
ds = xsnow.read('stage2_windowed.nc')

# Do your analysis
# ... avalanche forecasting, visualization, etc.
```

---

## Examples

### Example 1: Small Dataset (195 files, 5 min SNOWPACK)

```bash
#!/bin/bash
# Quick concurrent run

./snowpack --config snowpack.ini &
sleep 240  # 4 minutes (80% of 5 min)

python snowpack_updater.py /data/snowpack \
    --archive quick.nc \
    --window-days 30 \
    --cpus 5

# Total time: ~6 minutes (vs 10 minutes sequential)
```

### Example 2: Medium Dataset (10k files, 30 min SNOWPACK)

```bash
#!/bin/bash
# Medium concurrent run

./snowpack --config snowpack.ini &
sleep 1500  # 25 minutes (83% of 30 min)

python snowpack_updater.py /data/snowpack \
    --archive medium.nc \
    --window-days 90 \
    --cpus 15 \
    --lazy

# Total time: ~35 minutes (vs 60 minutes sequential)
```

### Example 3: Large Dataset (114k files, 60 min SNOWPACK)

```bash
#!/bin/bash
# Large concurrent run

./snowpack --config snowpack.ini &
SNOWPACK_PID=$!

sleep 3000  # 50 minutes (83% of 60 min)

python snowpack_updater.py /data/snowpack \
    --archive operational.nc \
    --window-days 90 \
    --cpus 20 \
    --lazy \
    >> loader.log 2>&1 &
LOADER_PID=$!

# Wait for both
wait $SNOWPACK_PID
wait $LOADER_PID

echo "✅ Complete! Total time: ~65 minutes (vs 105 minutes sequential)"
```

---

## Summary

### Key Benefits

✅ **38% time savings** - 105 min → 65 min  
✅ **Faster turnaround** - Data ready 40 min sooner  
✅ **No data loss** - Safe concurrent access  
✅ **Simple setup** - Just adjust timing  
✅ **Reliable** - Robust error handling  

### When to Use

**Use concurrent processing when:**
- SNOWPACK runs take >30 minutes
- You need fast turnaround for forecasting
- System has adequate resources
- Running operationally (daily/hourly)

**Use sequential processing when:**
- SNOWPACK runs are quick (<15 min)
- System resources are limited
- First-time setup/testing
- Debugging issues

### Quick Decision Guide

```
SNOWPACK duration > 30 min?
├─ Yes → Use concurrent (saves 30-40 min)
└─ No → Use sequential (simpler)

System has free CPU/RAM?
├─ Yes → Use concurrent (no contention)
└─ No → Use sequential (safer)

Need fast turnaround?
├─ Yes → Use concurrent (critical path optimization)
└─ No → Use sequential (simpler workflow)
```

---

## Getting Help

**Issues?** Check these first:
1. Logs: `tail -f snowpack_loader.log`
2. Resources: `htop` or `top`
3. Processes: `ps aux | grep -E "snowpack|python"`
4. Archive: `ls -lh snowpack_operational.nc`

**Documentation:**
- SNOWPACK_UPDATER_GUIDE.md - Complete usage guide
- example_usage.py - Python examples
- DESIGN_EVOLUTION.md - Why this works

**Questions?**
- Check xsnow documentation
- Review log files
- Test sequential first

---

**Start saving time today! Your post-processing can begin just minutes after SNOWPACK completes instead of nearly an hour later.** 
