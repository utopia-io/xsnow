#!/bin/bash
#
# check_archive_status.sh
#
# Monitor SNOWPACK archive system health
# Run this to check if everything is working correctly
#
# Usage: ./check_archive_status.sh

set -euo pipefail

#=============================================================================
# CONFIGURATION
#=============================================================================

OPERATIONAL_ARCHIVE="/data/operational.nc"
ARCHIVE_DIR="/data/archives"
LOG_DIR="/var/log/snowpack-archive"
DATA_DIR="/data/snowpack"

# Thresholds
MAX_AGE_HOURS=6          # Operational archive should be <6 hours old
MIN_DISK_FREE_GB=50      # Minimum free space
EXPECTED_LOCATIONS=114300  # Expected number of locations

#=============================================================================
# COLORS
#=============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#=============================================================================
# FUNCTIONS
#=============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo " SNOWPACK Archive System Status Check"
    echo "========================================================================"
    echo ""
}

check_operational() {
    echo "📊 Operational Archive:"
    echo ""
    
    if [ ! -f "${OPERATIONAL_ARCHIVE}" ]; then
        echo -e "${RED}✗ FAIL: Operational archive not found${NC}"
        echo "  Expected: ${OPERATIONAL_ARCHIVE}"
        return 1
    fi
    
    # File age
    local file_age=$(($(date +%s) - $(stat -c %Y "${OPERATIONAL_ARCHIVE}")))
    local hours_old=$((file_age / 3600))
    local minutes_old=$(( (file_age % 3600) / 60 ))
    
    if [ ${hours_old} -gt ${MAX_AGE_HOURS} ]; then
        echo -e "${RED}✗ FAIL: Archive too old (${hours_old}h ${minutes_old}m)${NC}"
        echo "  Should be updated every 6 hours"
        return 1
    else
        echo -e "${GREEN}✓ Age: ${hours_old}h ${minutes_old}m${NC}"
    fi
    
    # File size
    local size=$(du -h "${OPERATIONAL_ARCHIVE}" | cut -f1)
    echo -e "${GREEN}✓ Size: ${size}${NC}"
    
    # Check contents with Python
    if command -v python3 &> /dev/null; then
        local nc_info=$(python3 << 'EOF'
import xarray as xr
import sys
try:
    ds = xr.open_dataset('/data/operational.nc')
    print(f"{len(ds.location)}|{len(ds.time)}|{ds.time.values[0]}|{ds.time.values[-1]}")
    ds.close()
except Exception as e:
    print(f"ERROR|{e}", file=sys.stderr)
    sys.exit(1)
EOF
)
        
        if [ $? -eq 0 ]; then
            IFS='|' read -r n_loc n_time time_start time_end <<< "${nc_info}"
            
            echo -e "${GREEN}✓ Locations: ${n_loc}${NC}"
            
            # Check if expected number of locations
            if [ "${n_loc}" -ne ${EXPECTED_LOCATIONS} ]; then
                echo -e "${YELLOW}⚠ WARNING: Expected ${EXPECTED_LOCATIONS} locations${NC}"
            fi
            
            echo -e "${GREEN}✓ Timesteps: ${n_time}${NC}"
            echo -e "${GREEN}✓ Range: ${time_start} to ${time_end}${NC}"
        else
            echo -e "${RED}✗ FAIL: Cannot read NetCDF file${NC}"
            return 1
        fi
    fi
    
    echo ""
    return 0
}

check_archives() {
    echo "📚 Research Archives:"
    echo ""
    
    if [ ! -d "${ARCHIVE_DIR}" ]; then
        echo -e "${YELLOW}⚠ Archive directory not found: ${ARCHIVE_DIR}${NC}"
        echo ""
        return 0
    fi
    
    local n_archives=$(find "${ARCHIVE_DIR}" -name "archive_*.nc" -type f | wc -l)
    
    if [ ${n_archives} -eq 0 ]; then
        echo -e "${YELLOW}⚠ No research archives found${NC}"
    else
        echo -e "${GREEN}✓ Found ${n_archives} archive(s)${NC}"
        echo ""
        
        # List archives
        find "${ARCHIVE_DIR}" -name "archive_*.nc" -type f -printf "%T+ %s %p\n" | \
            sort -r | head -5 | while read -r date size path; do
            local size_gb=$(echo "scale=1; ${size} / 1073741824" | bc)
            local filename=$(basename "${path}")
            echo "  ${filename}: ${size_gb} GB (${date})"
        done
        
        if [ ${n_archives} -gt 5 ]; then
            echo "  ... and $((n_archives - 5)) more"
        fi
    fi
    
    echo ""
    return 0
}

check_disk_space() {
    echo "💾 Disk Space:"
    echo ""
    
    local free_gb=$(df -BG /data | awk 'NR==2 {print $4}' | sed 's/G//')
    local used_pct=$(df -h /data | awk 'NR==2 {print $5}')
    
    if [ ${free_gb} -lt ${MIN_DISK_FREE_GB} ]; then
        echo -e "${RED}✗ FAIL: Only ${free_gb}GB free (need ${MIN_DISK_FREE_GB}GB)${NC}"
        echo "  Used: ${used_pct}"
        return 1
    else
        echo -e "${GREEN}✓ Free: ${free_gb}GB (${used_pct} used)${NC}"
    fi
    
    echo ""
    return 0
}

check_memory() {
    echo "🧠 Memory:"
    echo ""
    
    local free_gb=$(free -g | awk 'NR==2 {print $7}')
    local total_gb=$(free -g | awk 'NR==2 {print $2}')
    local used_pct=$(echo "scale=0; ($total_gb - $free_gb) * 100 / $total_gb" | bc)
    
    echo -e "${GREEN}✓ Free: ${free_gb}GB / ${total_gb}GB (${used_pct}% used)${NC}"
    
    if [ ${free_gb} -lt 40 ]; then
        echo -e "${YELLOW}⚠ WARNING: Less than 40GB free${NC}"
        echo "  Archive updates need ~30GB"
    fi
    
    echo ""
    return 0
}

check_recent_logs() {
    echo "📝 Recent Activity:"
    echo ""
    
    if [ -f "${LOG_DIR}/operational.log" ]; then
        local last_success=$(grep "UPDATE COMPLETE" "${LOG_DIR}/operational.log" | tail -1)
        local last_error=$(grep -i "error\|fail" "${LOG_DIR}/operational.log" | tail -1)
        
        if [ -n "${last_success}" ]; then
            echo "  Last successful update:"
            echo "  ${last_success}"
        fi
        
        if [ -n "${last_error}" ]; then
            echo ""
            echo -e "${YELLOW}  Last error:${NC}"
            echo "  ${last_error}"
        fi
    else
        echo -e "${YELLOW}⚠ No operational logs found${NC}"
    fi
    
    echo ""
    return 0
}

check_snowpack_data() {
    echo "❄️  SNOWPACK Data:"
    echo ""
    
    if [ ! -d "${DATA_DIR}" ]; then
        echo -e "${RED}✗ FAIL: Data directory not found: ${DATA_DIR}${NC}"
        return 1
    fi
    
    # Find newest .pro file
    local newest=$(find "${DATA_DIR}" -name "*.pro" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
    
    if [ -z "${newest}" ]; then
        echo -e "${RED}✗ FAIL: No .pro files found${NC}"
        return 1
    fi
    
    local newest_file=$(echo "${newest}" | cut -d' ' -f2-)
    local file_age=$(($(date +%s) - $(stat -c %Y "${newest_file}")))
    local hours_old=$((file_age / 3600))
    local minutes_old=$(( (file_age % 3600) / 60 ))
    
    if [ ${hours_old} -gt 6 ]; then
        echo -e "${YELLOW}⚠ WARNING: SNOWPACK data is ${hours_old}h ${minutes_old}m old${NC}"
        echo "  SNOWPACK may not be running"
    else
        echo -e "${GREEN}✓ Data age: ${hours_old}h ${minutes_old}m${NC}"
    fi
    
    echo ""
    return 0
}

check_cron() {
    echo "⏰ Scheduled Jobs:"
    echo ""
    
    if crontab -l 2>/dev/null | grep -q "operational_archive\|run_operational_update"; then
        echo -e "${GREEN}✓ Operational update job found in crontab${NC}"
    else
        echo -e "${YELLOW}⚠ WARNING: No operational update job in crontab${NC}"
    fi
    
    if crontab -l 2>/dev/null | grep -q "snowpack_archiver\|run_archive_creation"; then
        echo -e "${GREEN}✓ Archive creation job found in crontab${NC}"
    else
        echo -e "${YELLOW}⚠ WARNING: No archive creation job in crontab${NC}"
    fi
    
    echo ""
    return 0
}

print_summary() {
    echo "========================================================================"
    echo " Summary"
    echo "========================================================================"
    echo ""
    
    if [ ${FAIL_COUNT} -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed!${NC}"
        echo ""
        echo "System is healthy and operational."
    else
        echo -e "${RED}✗ ${FAIL_COUNT} check(s) failed${NC}"
        echo ""
        echo "Please review errors above and take corrective action."
    fi
    
    echo ""
}

#=============================================================================
# MAIN
#=============================================================================

FAIL_COUNT=0

print_header

check_operational || ((FAIL_COUNT++))
check_snowpack_data || ((FAIL_COUNT++))
check_disk_space || ((FAIL_COUNT++))
check_memory || ((FAIL_COUNT++))
check_archives || ((FAIL_COUNT++))
check_recent_logs || ((FAIL_COUNT++))
check_cron || ((FAIL_COUNT++))

print_summary

exit ${FAIL_COUNT}
