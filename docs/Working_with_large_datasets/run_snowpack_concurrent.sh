#!/bin/bash
#
# run_snowpack_concurrent.sh - Run SNOWPACK and xsnow loader concurrently
#
# This script starts SNOWPACK, waits for it to be ~80% complete, then
# starts the xsnow data loader in parallel. This overlaps processing time
# and reduces total time by ~40 minutes for large datasets.
#
# Usage:
#   ./run_snowpack_concurrent.sh
#
# Requirements:
#   - SNOWPACK installed and configured
#   - snowpack_updater.py in current directory or PATH
#   - Sufficient CPU and RAM for both processes
#

set -e  # Exit on error

#==============================================================================
# CONFIGURATION - Adjust these for your system
#==============================================================================

# SNOWPACK Configuration
SNOWPACK_DIR="/home/snowpack"
SNOWPACK_CONFIG="snowpack.ini"
SNOWPACK_EXPECTED_DURATION=3600  # seconds (60 minutes)

# Data Paths
DATA_DIR="path/to/snowpack/output/files"
ARCHIVE_FILE="path_to_fast_loading_file.nc"

# Loader Configuration  
WINDOW_DAYS=14
LOADER_CPUS=15
USE_LAZY="--lazy"  # Use --eager only if data fits in RAM

# Timing
CONCURRENT_DELAY=$((SNOWPACK_EXPECTED_DURATION * 5 / 6))  # 83% of duration
echo "Calculated delay: $CONCURRENT_DELAY seconds (~$((CONCURRENT_DELAY/60)) minutes)"

# Logging
SNOWPACK_LOG="snowpack_run.log"
LOADER_LOG="snowpack_loader.log"
MAIN_LOG="concurrent_run.log"

#==============================================================================
# FUNCTIONS
#==============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check SNOWPACK directory
    if [ ! -d "$SNOWPACK_DIR" ]; then
        log "ERROR: SNOWPACK directory not found: $SNOWPACK_DIR"
        exit 1
    fi
    
    # Check SNOWPACK config
    if [ ! -f "$SNOWPACK_DIR/$SNOWPACK_CONFIG" ]; then
        log "ERROR: SNOWPACK config not found: $SNOWPACK_DIR/$SNOWPACK_CONFIG"
        exit 1
    fi
    
    # Check data directory
    if [ ! -d "$DATA_DIR" ]; then
        log "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Check for snowpack_updater.py
    if ! command -v python &> /dev/null; then
        log "ERROR: Python not found in PATH"
        exit 1
    fi
    
    if [ ! -f "snowpack_updater.py" ]; then
        log "ERROR: snowpack_updater.py not found in current directory"
        log "       Please ensure snowpack_updater.py is in: $(pwd)"
        exit 1
    fi
    
    log "✅ All prerequisites met"
}

check_resources() {
    log "Checking system resources..."
    
    # Check available memory (in GB)
    local available_mem_gb=$(free -g | awk '/^Mem:/{print $7}')
    log "Available memory: ${available_mem_gb} GB"
    
    if [ "$available_mem_gb" -lt 10 ]; then
        log "⚠️  WARNING: Low memory (<10 GB available)"
        log "   Consider using --lazy mode and reducing --cpus"
    fi
    
    # Check CPU count
    local cpu_count=$(nproc)
    log "CPU cores: $cpu_count"
    
    local total_cpus=$((4 + LOADER_CPUS))  # Assume SNOWPACK uses 4
    if [ "$total_cpus" -gt "$cpu_count" ]; then
        log "⚠️  WARNING: Requested CPUs ($total_cpus) > available ($cpu_count)"
        log "   SNOWPACK (~4) + Loader ($LOADER_CPUS) may overload system"
    fi
    
    log "✅ Resource check complete"
}

start_snowpack() {
    log "========================================"
    log "Starting SNOWPACK..."
    log "========================================"
    log "Directory: $SNOWPACK_DIR"
    log "Config: $SNOWPACK_CONFIG"
    log "Log: $SNOWPACK_LOG"
    
    cd "$SNOWPACK_DIR"
    ./snowpack --config "$SNOWPACK_CONFIG" > "../$SNOWPACK_LOG" 2>&1 &
    SNOWPACK_PID=$!
    cd - > /dev/null
    
    if ! ps -p $SNOWPACK_PID > /dev/null; then
        log "❌ ERROR: SNOWPACK failed to start"
        log "   Check $SNOWPACK_LOG for details"
        exit 1
    fi
    
    log "✅ SNOWPACK started (PID: $SNOWPACK_PID)"
    log "   Expected duration: $((SNOWPACK_EXPECTED_DURATION/60)) minutes"
    log "   Loader will start in: $((CONCURRENT_DELAY/60)) minutes"
}

wait_for_concurrent_start() {
    log ""
    log "⏳ Waiting $((CONCURRENT_DELAY/60)) minutes before starting loader..."
    log "   (Letting SNOWPACK process most files first)"
    
    local elapsed=0
    local interval=60  # Check every minute
    
    while [ $elapsed -lt $CONCURRENT_DELAY ]; do
        sleep $interval
        elapsed=$((elapsed + interval))
        
        # Check if SNOWPACK is still running
        if ! ps -p $SNOWPACK_PID > /dev/null; then
            log "⚠️  SNOWPACK finished early (after $((elapsed/60)) minutes)"
            log "   Starting loader now..."
            return 0
        fi
        
        # Progress indicator
        local progress=$((elapsed * 100 / CONCURRENT_DELAY))
        log "   Progress: ${progress}% ($((elapsed/60))/$((CONCURRENT_DELAY/60)) min) - SNOWPACK still running"
    done
    
    log "✅ Concurrent delay complete"
}

start_loader() {
    log ""
    log "========================================"
    log "Starting SnowpackUpdater (concurrent)..."
    log "========================================"
    log "Data directory: $DATA_DIR"
    log "Archive: $ARCHIVE_FILE"
    log "Window: $WINDOW_DAYS days"
    log "CPUs: $LOADER_CPUS"
    log "Mode: $USE_LAZY"
    log "Log: $LOADER_LOG"
    
    python snowpack_updater.py "$DATA_DIR" \
        --archive "$ARCHIVE_FILE" \
        --window-days $WINDOW_DAYS \
        --cpus $LOADER_CPUS \
        $USE_LAZY \
        > "$LOADER_LOG" 2>&1 &
    LOADER_PID=$!
    
    if ! ps -p $LOADER_PID > /dev/null; then
        log "❌ ERROR: Loader failed to start"
        log "   Check $LOADER_LOG for details"
        # SNOWPACK is still running, let it finish
        return 1
    fi
    
    log "✅ Loader started (PID: $LOADER_PID)"
    log ""
    log "Both processes now running concurrently:"
    log "  SNOWPACK: PID $SNOWPACK_PID (finishing up)"
    log "  Loader:   PID $LOADER_PID (loading ready files)"
}

wait_for_completion() {
    log ""
    log "⏳ Waiting for processes to complete..."
    log "   (You can monitor with: tail -f $SNOWPACK_LOG and tail -f $LOADER_LOG)"
    log ""
    
    # Wait for SNOWPACK
    if ps -p $SNOWPACK_PID > /dev/null; then
        log "   Waiting for SNOWPACK (PID $SNOWPACK_PID)..."
        wait $SNOWPACK_PID
        SNOWPACK_EXIT=$?
        
        if [ $SNOWPACK_EXIT -eq 0 ]; then
            log "✅ SNOWPACK completed successfully"
        else
            log "❌ SNOWPACK exited with error code: $SNOWPACK_EXIT"
            log "   Check $SNOWPACK_LOG for details"
        fi
    else
        log "✅ SNOWPACK already completed"
        SNOWPACK_EXIT=0
    fi
    
    # Wait for Loader
    if ps -p $LOADER_PID > /dev/null; then
        log "   Waiting for Loader (PID $LOADER_PID)..."
        wait $LOADER_PID
        LOADER_EXIT=$?
        
        if [ $LOADER_EXIT -eq 0 ]; then
            log "✅ Loader completed successfully"
        else
            log "❌ Loader exited with error code: $LOADER_EXIT"
            log "   Check $LOADER_LOG for details"
        fi
    else
        log "✅ Loader already completed"
        LOADER_EXIT=0
    fi
    
    return $((SNOWPACK_EXIT + LOADER_EXIT))
}

show_summary() {
    local exit_code=$1
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local duration_min=$((duration / 60))
    
    log ""
    log "========================================"
    log "CONCURRENT RUN COMPLETE"
    log "========================================"
    log "Total time: ${duration_min} minutes ($duration seconds)"
    log ""
    log "Files created:"
    
    if [ -f "$ARCHIVE_FILE" ]; then
        local size=$(du -h "$ARCHIVE_FILE" | cut -f1)
        log "  ✅ Archive: $ARCHIVE_FILE ($size)"
    else
        log "  ❌ Archive: $ARCHIVE_FILE (NOT FOUND)"
    fi
    
    log ""
    log "Logs:"
    log "  SNOWPACK: $SNOWPACK_LOG"
    log "  Loader:   $LOADER_LOG"
    log "  Main:     $MAIN_LOG"
    
    if [ $exit_code -eq 0 ]; then
        log ""
        log "🎉 SUCCESS! Data ready for analysis."
        log ""
        log "Time savings estimate:"
        log "  Sequential: ~$((SNOWPACK_EXPECTED_DURATION/60 + 45)) minutes"
        log "  Concurrent: ${duration_min} minutes"
        log "  Saved: ~$((SNOWPACK_EXPECTED_DURATION/60 + 45 - duration_min)) minutes"
    else
        log ""
        log "⚠️  Completed with errors. Check logs for details."
    fi
    
    log "========================================"
}

cleanup() {
    log ""
    log "🛑 Script interrupted! Cleaning up..."
    
    # Kill child processes if still running
    if ps -p $SNOWPACK_PID > /dev/null 2>&1; then
        log "   Stopping SNOWPACK (PID $SNOWPACK_PID)..."
        kill $SNOWPACK_PID
    fi
    
    if ps -p $LOADER_PID > /dev/null 2>&1; then
        log "   Stopping Loader (PID $LOADER_PID)..."
        kill $LOADER_PID
    fi
    
    log "Cleanup complete"
    exit 130
}

#==============================================================================
# MAIN
#==============================================================================

# Record start time
START_TIME=$(date +%s)

# Set up signal handlers
trap cleanup INT TERM

log "========================================"
log "CONCURRENT SNOWPACK + xsnow PROCESSING"
log "========================================"
log "Started: $(date)"
log ""

# Run the workflow
check_prerequisites
check_resources
start_snowpack
wait_for_concurrent_start
start_loader
wait_for_completion
EXIT_CODE=$?

# Show summary
show_summary $EXIT_CODE

exit $EXIT_CODE
