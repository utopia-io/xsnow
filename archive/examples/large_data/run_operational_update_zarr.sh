#!/bin/bash
#
# run_operational_update.sh - ZARR VERSION (FIXED)
# 
# Production wrapper for SNOWPACK operational archive updates using Zarr format
# Handles safety checks, logging, locking, and error notifications
#
# Usage: ./run_operational_update_zarr.sh /data/snowpack

set -euo pipefail  # Exit on error, undefined vars, pipe failures

#=============================================================================
# CONFIGURATION
#=============================================================================

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/operational_archive_zarr.py"
LOG_DIR="./logs/snowpack-archive"
LOCK_FILE="./snowpack-archive.lock"

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

# Archive settings (ZARR FORMAT)
WINDOW_DAYS=14

# Archive naming mode
# - "rolling": Use same filename (operational.zarr) - gets updated each run
# - "timestamped": Create dated archives (operational_2024-12-29_14:00-2024-12-29_20:00.zarr)
ARCHIVE_NAMING="timestamped"           # Options: "rolling" or "timestamped"
ARCHIVE_BASENAME="operational"     # Base name for archive

# Base archive path (will be modified if timestamped mode)
ARCHIVE_FILE="./data/${ARCHIVE_BASENAME}.zarr"

# Process priority (for concurrent execution with SNOWPACK)
# Lower priority = less interference with SNOWPACK
NICE_LEVEL=19           # CPU priority (0=normal, 19=lowest)
IONICE_CLASS=2          # I/O class (2=best-effort)
IONICE_LEVEL=7          # I/O priority (0=highest, 7=lowest)

# Resource limits
MIN_DISK_GB=50        # Minimum free disk space (GB)
MIN_MEMORY_GB=40      # Minimum free memory (GB)
MAX_RUNTIME_HOURS=3   # Kill if exceeds this

# Notifications (optional)
ENABLE_EMAIL=false
EMAIL_TO="your-email@example.com"
ENABLE_SLACK=false
SLACK_WEBHOOK=""

#=============================================================================
# FUNCTIONS
#=============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/wrapper.log"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_DIR}/wrapper.log" >&2
}

send_notification() {
    local subject="$1"
    local message="$2"
    
    # Email notification
    if [ "${ENABLE_EMAIL}" = true ]; then
        echo "${message}" | mail -s "${subject}" "${EMAIL_TO}"
    fi
    
    # Slack notification
    if [ "${ENABLE_SLACK}" = true ] && [ -n "${SLACK_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${subject}: ${message}\"}" \
            "${SLACK_WEBHOOK}" 2>/dev/null || true
    fi
}

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

check_memory() {
    local free_gb=$(free -g | awk 'NR==2 {print $7}')
    
    if [ "${free_gb}" -lt "${MIN_MEMORY_GB}" ]; then
        log_error "Insufficient memory: ${free_gb}GB free, need ${MIN_MEMORY_GB}GB"
        return 1
    fi
    
    log "Memory OK: ${free_gb}GB free"
    return 0
}

check_snowpack_running() {
    # Check if SNOWPACK is running (informational only)
    # We run CONCURRENTLY with SNOWPACK, not after it!
    if pgrep -f "snowpack" > /dev/null; then
        log "SNOWPACK is running - will process files concurrently"
        return 0
    else
        log "SNOWPACK not running - processing finalized files"
        return 0
    fi
}

get_priority_command() {
    # Build priority command prefix
    # Uses nice for CPU priority, ionice for I/O priority (if available)
    
    local cmd="nice -n ${NICE_LEVEL}"
    
    # Check if ionice is available
    if command -v ionice >/dev/null 2>&1; then
        cmd="${cmd} ionice -c ${IONICE_CLASS} -n ${IONICE_LEVEL}"
    fi
    
    echo "${cmd}"
}

get_archive_filename() {
    # Generate archive filename based on naming mode
    # IMPORTANT: This function ONLY outputs the filename to stdout
    # NO log() calls allowed here - they pollute stdout!
    
    local data_dir="$1"
    
    if [ "${ARCHIVE_NAMING}" = "timestamped" ]; then
        # Find most recent .pro file to determine SNOWPACK end time
        local newest_file=$(find "${data_dir}" -name "*.pro" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        
        if [ -z "${newest_file}" ]; then
            # Fallback to rolling mode if can't determine SNOWPACK runtime
            echo "./data/${ARCHIVE_BASENAME}.zarr"
            return
        fi
        
        # Get modification time of newest file (SNOWPACK end time)
        local end_timestamp=$(stat -c %Y "${newest_file}" 2>/dev/null)
        if [ -z "${end_timestamp}" ]; then
            echo "./data/${ARCHIVE_BASENAME}.zarr"
            return
        fi
        
        local end_date=$(date -d "@${end_timestamp}" +"%Y-%m-%d_%H:%M" 2>/dev/null)
        
        # Calculate start time (end - window_days)
        local start_timestamp=$((end_timestamp - WINDOW_DAYS * 86400))
        local start_date=$(date -d "@${start_timestamp}" +"%Y-%m-%d_%H:%M" 2>/dev/null)
        
        local timestamped_file="./data/${ARCHIVE_BASENAME}_${start_date}_${end_date}.zarr"
        
        # Output ONLY the filename - nothing else!
        echo "${timestamped_file}"
    else
        # Rolling mode - same filename
        echo "${ARCHIVE_FILE}"
    fi
}

check_recent_snowpack_run() {
    local data_dir="$1"
    
    # Find most recent .pro file modification
    local newest_file=$(find "${data_dir}" -name "*.pro" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "${newest_file}" ]; then
        log_error "No .pro files found in ${data_dir}"
        return 1
    fi
    
    local file_age=$(($(date +%s) - $(stat -c %Y "${newest_file}")))
    local hours_old=$((file_age / 3600))
    local minutes_old=$(( (file_age % 3600) / 60 ))
    
    if [ "${hours_old}" -gt 24 ]; then
        log_error "SNOWPACK data is ${hours_old} hours old - SNOWPACK may not be running"
        return 1
    fi
    
    if [ "${file_age}" -lt 600 ]; then
        log "SNOWPACK data very recent (${minutes_old} min old) - likely running concurrently"
    else
        log "SNOWPACK data is ${hours_old}h ${minutes_old}m old"
    fi
    
    return 0
}

acquire_lock() {
    if [ -e "${LOCK_FILE}" ]; then
        local lock_pid=$(cat "${LOCK_FILE}" 2>/dev/null || echo "unknown")
        
        # Check if process is still running
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

release_lock() {
    rm -f "${LOCK_FILE}"
    log "Released lock"
}

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
        send_notification "SNOWPACK Archive Update FAILED" \
            "Update failed with exit code ${exit_code}. Check ${LOG_DIR}/wrapper.log"
    fi
    
    exit ${exit_code}
}

#=============================================================================
# MAIN
#=============================================================================

# Trap signals for cleanup
trap cleanup EXIT INT TERM

# Check arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <snowpack_data_dir>"
    exit 1
fi

DATA_DIR="$1"

# Create log directory FIRST
mkdir -p "${LOG_DIR}"

# Determine archive filename (capture function output cleanly)
# Redirect stderr to /dev/null to avoid pollution
ARCHIVE_FILE=$(get_archive_filename "${DATA_DIR}" 2>/dev/null)

# Create data directory for archive file
mkdir -p "$(dirname "${ARCHIVE_FILE}")"

# Now log everything (after filename is determined)
log "========================================================================"
log "SNOWPACK Operational Archive Update - ZARR VERSION"
log "========================================================================"
log "Data directory: ${DATA_DIR}"
log "Archive file: ${ARCHIVE_FILE}"
log "Window: ${WINDOW_DAYS} days"
log "Naming mode: ${ARCHIVE_NAMING}"

# If timestamped, show the time range
if [ "${ARCHIVE_NAMING}" = "timestamped" ]; then
    # Extract dates from filename for display
    if [[ "${ARCHIVE_FILE}" =~ _([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2})_([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}) ]]; then
        log "Time range: ${BASH_REMATCH[1]} to ${BASH_REMATCH[2]}"
    fi
fi

log ""

# Pre-flight checks
log "Running pre-flight checks..."

if ! check_disk_space "$(dirname "${ARCHIVE_FILE}")"; then
    exit 1
fi

if ! check_memory; then
    exit 1
fi

# Check if SNOWPACK is running (informational)
log "Checking SNOWPACK status..."
check_snowpack_running

if ! check_recent_snowpack_run "${DATA_DIR}"; then
    log_error "SNOWPACK data not current - skipping update"
    exit 1
fi

if ! acquire_lock; then
    exit 1
fi

# Verify Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    log_error "Python script not found: ${PYTHON_SCRIPT}"
    log_error ""
    log_error "Please copy operational_archive_zarr.py to: ${SCRIPT_DIR}/"
    log_error "Or update PYTHON_SCRIPT variable in this script"
    exit 1
fi

log ""
log "All checks passed - starting update"
log ""

# Run the update with timeout
START_TIME=$(date +%s)

# Get priority command (nice and ionice if available)
PRIORITY_CMD=$(get_priority_command)
log "Using priority command: ${PRIORITY_CMD}"

# Execute with low priority for concurrent operation
${PRIORITY_CMD} timeout ${MAX_RUNTIME_HOURS}h "${PYTHON_EXE}" "${PYTHON_SCRIPT}" \
    "${DATA_DIR}" \
    --archive "${ARCHIVE_FILE}" \
    --window-days ${WINDOW_DAYS} \
    >> "${LOG_DIR}/operational.log" 2>&1 &

UPDATE_PID=$!
log "Update started (PID: ${UPDATE_PID})"

# Wait for completion
wait ${UPDATE_PID}
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

log ""
log "========================================================================"

if [ ${EXIT_CODE} -eq 0 ]; then
    log "✅ UPDATE COMPLETE"
    log "Duration: ${DURATION_MIN} minutes"
    
    # Verify archive was updated
    if [ -e "${ARCHIVE_FILE}" ]; then
        ARCHIVE_SIZE=$(du -sh "${ARCHIVE_FILE}" | cut -f1)
        
        log "Archive file: ${ARCHIVE_FILE}"
        log "Archive size: ${ARCHIVE_SIZE}"
    else
        log_error "WARNING: Archive file not found!"
    fi
    
else
    log_error "❌ UPDATE FAILED"
    log_error "Duration: ${DURATION_MIN} minutes"
    log_error "Exit code: ${EXIT_CODE}"
    
    if [ ${EXIT_CODE} -eq 124 ]; then
        log_error "Update exceeded ${MAX_RUNTIME_HOURS} hour timeout"
    elif [ ${EXIT_CODE} -eq 127 ]; then
        log_error "Command not found - check Python script path"
    fi
fi

log "========================================================================"
log ""

exit ${EXIT_CODE}