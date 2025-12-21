#!/bin/bash
#
# run_archive_creation.sh
#
# Production wrapper for monthly SNOWPACK archive creation
# Creates clean, consolidated archives for long-term storage and research
#
# Usage: ./run_archive_creation.sh /data/snowpack [YYYY_MM]

set -euo pipefail

#=============================================================================
# CONFIGURATION
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/snowpack_archiver.py"
LOG_DIR="/var/log/snowpack-archive"
ARCHIVE_DIR="/data/archives"

# Archive settings
WINDOW_DAYS=365
MIN_DISK_GB=100  # Need more space for large archives

# Notifications
ENABLE_EMAIL=false
EMAIL_TO="your-email@example.com"

#=============================================================================
# FUNCTIONS
#=============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/archive.log"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_DIR}/archive.log" >&2
}

send_notification() {
    if [ "${ENABLE_EMAIL}" = true ]; then
        echo "$2" | mail -s "$1" "${EMAIL_TO}"
    fi
}

check_disk_space() {
    local free_gb=$(df -BG "${ARCHIVE_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "${free_gb}" -lt "${MIN_DISK_GB}" ]; then
        log_error "Insufficient disk space: ${free_gb}GB free, need ${MIN_DISK_GB}GB"
        return 1
    fi
    
    log "Disk space OK: ${free_gb}GB free"
    return 0
}

#=============================================================================
# MAIN
#=============================================================================

if [ $# -lt 1 ]; then
    log_error "Usage: $0 <snowpack_data_dir> [YYYY_MM]"
    exit 1
fi

DATA_DIR="$1"
MONTH="${2:-$(date +%Y_%m)}"  # Default to current month

OUTPUT_FILE="${ARCHIVE_DIR}/archive_${MONTH}.nc"

mkdir -p "${LOG_DIR}"
mkdir -p "${ARCHIVE_DIR}"

log "========================================================================"
log "SNOWPACK Archive Creation - Starting"
log "========================================================================"
log "Data directory: ${DATA_DIR}"
log "Output file: ${OUTPUT_FILE}"
log "Window: ${WINDOW_DAYS} days"
log ""

# Check if archive already exists
if [ -f "${OUTPUT_FILE}" ]; then
    log "WARNING: Archive already exists: ${OUTPUT_FILE}"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Aborted by user"
        exit 0
    fi
    log "Will overwrite existing archive"
fi

# Pre-flight checks
if ! check_disk_space; then
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    log_error "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

log ""
log "Starting archive creation (this will take 3-4 hours)..."
log ""

START_TIME=$(date +%s)

python3 "${PYTHON_SCRIPT}" \
    "${DATA_DIR}" \
    --window-days ${WINDOW_DAYS} \
    --output "${OUTPUT_FILE}" \
    >> "${LOG_DIR}/archive.log" 2>&1

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_HOURS=$((DURATION / 3600))
DURATION_MIN=$(( (DURATION % 3600) / 60 ))

log ""
log "========================================================================"

if [ ${EXIT_CODE} -eq 0 ]; then
    log "✅ ARCHIVE COMPLETE"
    log "Duration: ${DURATION_HOURS}h ${DURATION_MIN}m"
    
    if [ -f "${OUTPUT_FILE}" ]; then
        ARCHIVE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        log "Archive: ${OUTPUT_FILE}"
        log "Size: ${ARCHIVE_SIZE}"
        
        send_notification "SNOWPACK Archive Created" \
            "Successfully created ${OUTPUT_FILE} (${ARCHIVE_SIZE}) in ${DURATION_HOURS}h ${DURATION_MIN}m"
    fi
else
    log_error "❌ ARCHIVE FAILED"
    log_error "Exit code: ${EXIT_CODE}"
    
    send_notification "SNOWPACK Archive FAILED" \
        "Archive creation failed with exit code ${EXIT_CODE}. Check ${LOG_DIR}/archive.log"
fi

log "========================================================================"

exit ${EXIT_CODE}
