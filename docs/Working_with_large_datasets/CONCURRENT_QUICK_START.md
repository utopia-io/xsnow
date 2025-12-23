# Concurrent Processing - Quick Start Guide

**Get your xsnow data ready 40 minutes faster by running the loader while SNOWPACK is still running!**

---

## 5-Minute Setup

### Step 1: Get the Files

You need these files:
- ✅ `snowpack_updater.py` (main script)
- ✅ `run_snowpack_concurrent.sh` (wrapper script)

```bash
cd /home/ron/xsnow/examples/largescale

# Verify files exist
ls -l snowpack_updater.py run_snowpack_concurrent.sh

# Make script executable
chmod +x run_snowpack_concurrent.sh
```

### Step 2: Configure the Script

Edit `run_snowpack_concurrent.sh` with your paths:

```bash
nano run_snowpack_concurrent.sh
```

**Change these lines:**
```bash

# SNOWPACK Configuration
SNOWPACK_DIR="/home/snowpack"                    # Your SNOWPACK directory
SNOWPACK_CONFIG="snowpack.ini"                  # Your config file
SNOWPACK_EXPECTED_DURATION=3600                 # 60 minutes (adjust!)

# Data Paths
DATA_DIR="/home/snowpack/runs/output.2024"     # Where .pro/.smet files are
ARCHIVE_FILE="snowpack_operational.nc"          # Output file name

# Loader Configuration  
WINDOW_DAYS=90                                   # Keep last N days
LOADER_CPUS=15                                   # CPUs for loader
```

**Critical:** Set `SNOWPACK_EXPECTED_DURATION` correctly!
- Measure: Run SNOWPACK once, time it
- Set to: That duration in seconds
- Example: 60 min → 3600 seconds

### Step 3: Test Run

```bash
# First time: watch it carefully
./run_snowpack_concurrent.sh

# Monitor in real-time (open 3 terminals):
# Terminal 1:
tail -f snowpack_run.log

# Terminal 2:
tail -f snowpack_loader.log

# Terminal 3:
watch -n 5 'ps aux | grep -E "snowpack|python" | grep -v grep'
```

### Step 4: Verify Results

After completion:

```bash
# Check archive was created
ls -lh snowpack_operational.nc

# Quick data check
python -c "
import xsnow
ds = xsnow.read('snowpack_operational.nc')
print(f'Locations: {len(ds.data.location):,}')
print(f'Time points: {len(ds.data.time):,}')
print(f'Date range: {ds.data.time.values[0]} to {ds.data.time.values[-1]}')
"
```

### Step 5: Automate (Optional)

Add to crontab for automatic runs:

```bash
crontab -e
```

Add this line:
```bash
# Run at 2 AM daily (adjust as needed)
0 2 * * * cd /your/working/path && ./run_snowpack_concurrent.sh >> /var/log/snowpack_concurrent.log 2>&1
```

---

## How It Works (Simple Version)

```
1. SNOWPACK starts
   └─ Begins updating 114,300 files
   └─ Takes ~60 minutes

2. Wait 50 minutes
   └─ Let SNOWPACK process most files

3. Start Loader (while SNOWPACK finishes)
   └─ Loads files that are ready
   └─ Takes ~15 minutes

4. SNOWPACK finishes (60 min mark)
   └─ All files updated

5. Loader finishes (65 min mark)
   └─ Archive ready!

Total: 65 minutes
vs Sequential: 105 minutes
Saved: 40 minutes! 
```

---

## Troubleshooting

### Problem: Loader starts too early

**Symptom:** Archive missing latest timesteps

**Fix:** Increase delay in script:
```bash
SNOWPACK_EXPECTED_DURATION=4200  # Was 3600, now 70 min
```

### Problem: Loader waits too long

**Symptom:** Both finish, then loader starts

**Fix:** Decrease delay:
```bash
SNOWPACK_EXPECTED_DURATION=3000  # Was 3600, now 50 min
```

### Problem: Out of memory

**Symptom:** Process killed, system slow

**Fix 1:** Reduce loader CPUs:
```bash
LOADER_CPUS=10  # Was 15
```

**Fix 2:** Ensure lazy mode:
```bash
USE_LAZY="--lazy"  # Make sure this is set!
```

### Problem: Script can't find snowpack_updater.py

**Symptom:** "snowpack_updater.py not found"

**Fix:** Put it in the same directory:
```bash
cd /your/working/path
ls snowpack_updater.py  # Should exist here
```

---

## Manual Method (Even Simpler)

Don't want to use the script? Do it manually:

### Terminal 1: Start SNOWPACK
```bash
cd /home/snowpack
./snowpack --config snowpack.ini
```

### Terminal 2: Wait, then start loader
```bash
# Wait for SNOWPACK to be ~80-90% done
# Then run:
cd /home/ron/xsnow/examples/largescale
python snowpack_updater.py /home/snowpack/runs/output \
    --archive snowpack_operational.nc \
    --window-days 14 \
    --cpus 15 \
    --lazy
```

**That's it!** Loader will finish ~5 min after SNOWPACK.

---

## Performance Comparison

### Your 114,300 Files (60 min SNOWPACK run)

**Sequential (old way):**
```
SNOWPACK:     60 min  ████████████████████
Wait:          0 min
Loader:       45 min  ████████████
─────────────────────────────────
Total:       105 min
```

**Concurrent (new way):**
```
SNOWPACK:     60 min  ████████████████████
Loader:       15 min          ████  (overlapped!)
─────────────────────────────────
Total:        65 min
```

**Time saved: 40 minutes (38% faster!)**

---

## When to Use

✅ **Use concurrent when:**
- SNOWPACK takes >30 minutes
- You need fast turnaround
- System has free CPU/RAM
- Running operationally

❌ **Use sequential when:**
- SNOWPACK takes <15 minutes
- Testing/debugging
- Limited resources
- First time setup

---

## Next Steps

1. ✅ Set up the script (5 min)
2. ✅ Test it once (watch logs)
3. ✅ Verify archive looks good
4. ✅ Automate with cron
5. ✅ Save 40 min every run!

---

## Getting Help

**Something not working?**

1. Check logs:
   - `tail -f concurrent_run.log` (main)
   - `tail -f snowpack_run.log` (SNOWPACK)
   - `tail -f snowpack_loader.log` (loader)

2. Check processes:
   ```bash
   ps aux | grep -E "snowpack|python"
   ```

3. Check resources:
   ```bash
   free -h  # Memory
   df -h    # Disk space
   htop     # CPU usage
   ```

4. Read the full guide:
   - `CONCURRENT_PROCESSING_README.md`

**Questions?**
- Review example output in logs
- Check SNOWPACK_UPDATER_GUIDE.md
- Test sequential first to verify basics work

---

## Summary

**What you need:**
- 2 files: `snowpack_updater.py` + `run_snowpack_concurrent.sh`
- 5 minutes to configure
- First test run to verify

**What you get:**
- 40 minutes saved per run
- Data ready right after SNOWPACK
- Same quality results
- Automated workflow

**Start saving time today!** 

---

## Quick Reference

```bash
# Configure (once)
nano run_snowpack_concurrent.sh  # Set your paths

# Run
./run_snowpack_concurrent.sh

# Monitor
tail -f concurrent_run.log

# Verify
ls -lh snowpack_operational.nc
python -c "import xsnow; ds = xsnow.read('snowpack_operational.nc'); print(ds.data)"

# Automate
crontab -e  # Add scheduled run
```

**That's all you need to know!** 
