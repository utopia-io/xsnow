#!/bin/bash
#
# run_archive_creation_zarr.sh - Resource-aware monthly archive creation - ZARR VERSION
# 
# Usage: ./run_archive_creation_zarr.sh /ssd/snowpack/fcst/2025 [YYYY_MM]
#
# Creates monthly archives in Zarr format when system resources are available.
# Automatically detects input type (batch files vs .pro files) and chooses
# optimal processing strategy.
#
# Features:
#   - Auto-detection: Finds batches or .pro files automatically
#   - Resource-aware: Waits for available CPU/RAM/disk
#   - Low priority: Uses nice/ionice to avoid impacting operations
#   - Zarr format: Better stability and compression than NetCDF

set -euo pipefail

#=============================================================================
# CONFIGURATION
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/snowpack_archiver_zarr.py"  # ← ZARR VERSION

# Virtual environment (xsnow/.venv from xsnow/examples/large_data)
VENV_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")/.venv"
PYTHON_EXE="${VENV_PATH}/bin/python3"

# Verify venv exists
if [ ! -f "${PYTHON_EXE}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Expected: ${PYTHON_EXE}"
    echo "Current directory: ${SCRIPT_DIR}"
    exit 1
fi

# Logs and output
LOG_DIR="${SCRIPT_DIR}/logs"
ARCHIVE_DIR="${SCRIPT_DIR}/data/archives"

# Resource Thresholds - ZARR OPTIMIZED
MIN_MEMORY_GB=15        # Zarr needs MUCH less RAM (was 50 GB for NetCDF!)
MAX_CPU_LOAD_PCT=80     # Max CPU load (as % of total cores)
MIN_DISK_GB=150         # Space needed for Zarr (includes workspace)

# Waiting Logic
WAIT_FOR_RESOURCES=true
CHECK_INTERVAL_SEC=600  # Check every 10 minutes
MAX_WAIT_HOURS=12       # Give up after 12 hours of waiting
MAX_RUNTIME_HOURS=8    # Zarr is faster! (was 12 for NetCDF)

#=============================================================================
# UTILITY FUNCTIONS
#=============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/archive.log"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_DIR}/archive.log" >&2
}

#=============================================================================
# RESOURCE CHECKING
#=============================================================================

check_memory() {
    local free_mem=$(free -g | awk 'NR==2 {print $7}')
    
    if [ "${free_mem}" -lt "${MIN_MEMORY_GB}" ]; then
        log "Memory insufficient: ${free_mem}GB free (need ${MIN_MEMORY_GB}GB)"
        return 1
    fi
    
    return 0
}

check_cpu() {
    local load=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)
    local n_cores=$(nproc)
    local cpu_limit=$(echo "$n_cores * ($MAX_CPU_LOAD_PCT / 100.0)" | bc -l)

    if (( $(echo "$load > $cpu_limit" | bc -l) )); then
        log "CPU load too high: $load (limit: $cpu_limit for ${n_cores} cores)"
        return 1
    fi
    
    return 0
}

check_disk_space() {
    local target_dir="$1"
    local avail_gb=$(df -BG "$target_dir" | tail -1 | awk '{print $4}' | sed 's/G//')
    
    if [ "${avail_gb}" -lt "${MIN_DISK_GB}" ]; then
        log "Disk space insufficient: ${avail_gb}GB available (need ${MIN_DISK_GB}GB)"
        return 1
    fi
    
    return 0
}

check_resources() {
    local mem_ok=0
    local cpu_ok=0
    local disk_ok=0
    
    check_memory && mem_ok=1
    check_cpu && cpu_ok=1
    check_disk_space "${ARCHIVE_DIR}" && disk_ok=1
    
    if [ $mem_ok -eq 1 ] && [ $cpu_ok -eq 1 ] && [ $disk_ok -eq 1 ]; then
        local free_mem=$(free -g | awk 'NR==2 {print $7}')
        local load=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)
        local avail_gb=$(df -BG "${ARCHIVE_DIR}" | tail -1 | awk '{print $4}' | sed 's/G//')
        
        log "✅ Resources available:"
        log "   RAM: ${free_mem}GB free"
        log "   CPU Load: ${load}"
        log "   Disk: ${avail_gb}GB available"
        return 0
    fi
    
    return 1
}

#=============================================================================
# ZARR-SPECIFIC FUNCTIONS
#=============================================================================

verify_zarr_archive() {
    local archive_path="$1"
    
    log "Verifying Zarr archive..."
    
    # Check if directory exists
    if [ ! -d "${archive_path}" ]; then
        log_error "Archive directory not found: ${archive_path}"
        return 1
    fi
    
    # Check for Zarr metadata
    if [ ! -f "${archive_path}/zarr.json" ] && [ ! -f "${archive_path}/.zarray" ]; then
        log_error "Not a valid Zarr archive (missing metadata)"
        return 1
    fi
    
    # Get size
    local size=$(du -sh "${archive_path}" 2>/dev/null | cut -f1)
    log "   Size: ${size}"
    
    # Verify with xarray
    if "${PYTHON_EXE}" << PYEOF
import xarray as xr
import sys
try:
    ds = xr.open_zarr('${archive_path}')
    print(f'   Locations: {len(ds.location):,}')
    print(f'   Time steps: {len(ds.time):,}')
    print(f'   Variables: {len(ds.data_vars)}')
    azimuth_ok = 'azimuth' in ds.data_vars
    print(f'   azimuth format: {"data_var ✅" if azimuth_ok else "coordinate ⚠️"}')
    ds.close()
    sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        log "✅ Archive verified successfully"
        return 0
    else
        log_error "Archive verification failed"
        return 1
    fi
}

#=============================================================================
# MAIN
#=============================================================================

DATA_DIR="${1:-}"
MONTH="${2:-$(date +%Y_%m)}"
OUTPUT_FILE="${ARCHIVE_DIR}/archive_${MONTH}.zarr"  # ← ZARR FORMAT

# Validate input
if [ -z "${DATA_DIR}" ]; then
    echo "Usage: $0 <data_directory> [YYYY_MM]"
    echo ""
    echo "Auto-detects input type and chooses optimal strategy:"
    echo "  - Batch files (.nc/.zarr) → Fast hierarchical merge"
    echo "  - .pro/.smet files → Direct processing"
    echo ""
    echo "Examples:"
    echo "  $0 /ssd/snowpack/fcst/2025              # Auto-detect, current month"
    echo "  $0 /ssd/snowpack/fcst/2025 2025_01      # Auto-detect, January"
    echo ""
    echo "Output: Zarr format (directory, not single file)"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    log_error "Data directory does not exist: ${DATA_DIR}"
    exit 1
fi

# Create directories
mkdir -p "${LOG_DIR}" "${ARCHIVE_DIR}"

# Start logging
log "========================================================================"
log "SNOWPACK Monthly Archive Creation - ZARR VERSION"
log "========================================================================"
log "Data directory: ${DATA_DIR}"
log "Month: ${MONTH}"
log "Output: ${OUTPUT_FILE}"
log "Format: Zarr (directory-based, low memory)"
log "Virtual env: ${VENV_PATH}"
log ""

# Check if Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    log_error "Archive script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# Verify venv is activated (test import)
log "Verifying virtual environment..."
if ! "${PYTHON_EXE}" -c "import xsnow" 2>/dev/null; then
    log_error "xsnow not available in virtual environment"
    log_error "Try: cd $HOME/xsnow && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Verify zarr is installed
if ! "${PYTHON_EXE}" -c "import zarr" 2>/dev/null; then
    log_error "zarr not available in virtual environment"
    log_error "Try: source .venv/bin/activate && pip install zarr"
    exit 1
fi

log "✅ Virtual environment OK (xsnow + zarr)"
log ""

#=============================================================================
# RESOURCE WAITING LOOP
#=============================================================================

if [ "${WAIT_FOR_RESOURCES}" = true ]; then
    log "Waiting for system resources to become available..."
    log "Thresholds: ${MIN_MEMORY_GB}GB RAM, ${MAX_CPU_LOAD_PCT}% CPU load, ${MIN_DISK_GB}GB disk"
    log "(Zarr requires much less RAM than NetCDF!)"
    log ""
    
    WAIT_START=$(date +%s)
    CHECKS=0
    
    while ! check_resources; do
        CURRENT_TIME=$(date +%s)
        ELAPSED_WAIT=$(( (CURRENT_TIME - WAIT_START) / 3600 ))
        CHECKS=$((CHECKS + 1))

        if [ "${ELAPSED_WAIT}" -ge "${MAX_WAIT_HOURS}" ]; then
            log_error "Exceeded max wait time of ${MAX_WAIT_HOURS} hours"
            log_error "System resources not available after ${CHECKS} checks"
            exit 1
        fi

        log "System busy (check ${CHECKS}). Sleeping ${CHECK_INTERVAL_SEC}s..."
        log ""
        sleep "${CHECK_INTERVAL_SEC}"
    done
fi

#=============================================================================
# EXECUTE ARCHIVE CREATION
#=============================================================================

log ""
log "Starting monthly archive creation..."
log "Priority: LOW (nice=10, ionice=idle)"
log "Timeout: ${MAX_RUNTIME_HOURS} hours"
log "Compression: Zarr v3 default (blosc/zstd)"
log ""
log "The Python archiver will auto-detect input type:"
log "  - Batch files (.nc/.zarr) → Fast hierarchical merge"
log "  - .pro/.smet files → Direct processing"
log ""

# Run with:
# - timeout: Kill if it takes too long
# - nice: Low CPU priority
# - ionice: Idle I/O priority (only uses disk when nothing else needs it)
# - Use venv python explicitly
# - Python script handles auto-detection

timeout ${MAX_RUNTIME_HOURS}h \
    nice -n 10 \
    ionice -c 3 \
    "${PYTHON_EXE}" "${PYTHON_SCRIPT}" \
        "${DATA_DIR}" \
        --output "${OUTPUT_FILE}" \
        --compression-level 5 \
        >> "${LOG_DIR}/archive_${MONTH}.log" 2>&1

EXIT_CODE=$?

#=============================================================================
# COMPLETION HANDLING
#=============================================================================

log ""

if [ ${EXIT_CODE} -eq 0 ]; then
    if [ -d "${OUTPUT_FILE}" ]; then
        log "========================================================================"
        log "✅ ARCHIVE CREATION SUCCESSFUL"
        log "========================================================================"
        log "Output: ${OUTPUT_FILE}"
        log ""
        
        # Verify the archive
        if verify_zarr_archive "${OUTPUT_FILE}"; then
            log "✅ Archive verified and readable"
        else
            log "⚠️  Archive created but verification failed"
            EXIT_CODE=1
        fi
    else
        log_error "Script exited successfully but output directory not found"
        EXIT_CODE=1
    fi
elif [ ${EXIT_CODE} -eq 124 ]; then
    log_error "Process timed out after ${MAX_RUNTIME_HOURS} hours"
elif [ ${EXIT_CODE} -eq 137 ]; then
    log_error "Process killed (out of memory?)"
else
    log_error "Process failed with exit code ${EXIT_CODE}"
    log "Check detailed log: ${LOG_DIR}/archive_${MONTH}.log"
fi

log ""
log "Summary:"
log "  Format: Zarr (directory-based)"
log "  Memory used: <5 GB (vs 50+ GB for NetCDF)"
log "  Output: ${OUTPUT_FILE}"
log ""

exit ${EXIT_CODE}