#!/bin/bash
#
# run_operational_update.sh
# 
# Production wrapper for SNOWPACK operational archive updates
# Handles safety checks, logging, locking, and error notifications
#
# Usage: ./run_operational_update.sh /data/snowpack

set -euo pipefail  # Exit on error, undefined vars, pipe failures

#=============================================================================
# CONFIGURATION
#=============================================================================

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/operational_archive.py"
LOG_DIR="./log/snowpack-archive"
LOCK_FILE="./snowpack-archive.lock"

# Archive settings
ARCHIVE_FILE="./data/operational.nc"
WINDOW_DAYS=14

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

# Create log directory
mkdir -p "${LOG_DIR}"

# Create data directory for archive file
mkdir -p "$(dirname "${ARCHIVE_FILE}")"

log "========================================================================"
log "SNOWPACK Operational Archive Update - Starting"
log "========================================================================"
log "Data directory: ${DATA_DIR}"
log "Archive file: ${ARCHIVE_FILE}"
log "Window: ${WINDOW_DAYS} days"
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
    exit 1
fi

log ""
log "All checks passed - starting update"
log ""

# Run the update with timeout
START_TIME=$(date +%s)

timeout ${MAX_RUNTIME_HOURS}h python3 "${PYTHON_SCRIPT}" \
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
    if [ -f "${ARCHIVE_FILE}" ]; then
        ARCHIVE_AGE=$(($(date +%s) - $(stat -c %Y "${ARCHIVE_FILE}")))
        ARCHIVE_SIZE=$(du -h "${ARCHIVE_FILE}" | cut -f1)
        
        log "Archive file: ${ARCHIVE_FILE}"
        log "Archive size: ${ARCHIVE_SIZE}"
        log "Archive age: ${ARCHIVE_AGE} seconds"
        
        if [ ${ARCHIVE_AGE} -gt 600 ]; then
            log_error "WARNING: Archive file not recently updated!"
        fi
    else
        log_error "WARNING: Archive file not found!"
    fi
    
else
    log_error "❌ UPDATE FAILED"
    log_error "Duration: ${DURATION_MIN} minutes"
    log_error "Exit code: ${EXIT_CODE}"
    
    if [ ${EXIT_CODE} -eq 124 ]; then
        log_error "Update exceeded ${MAX_RUNTIME_HOURS} hour timeout"
    fi
fi

log "========================================================================"
log ""

exit ${EXIT_CODE}