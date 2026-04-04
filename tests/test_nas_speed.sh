#!/bin/bash

# Define mount points
DRIVE_B="/mnt/synology"

# Configuration: 2GB test file
FILE_NAME="io_test_file.tmp"
BS="1M"
COUNT="2000"

echo "=========================================="
echo "   I/O SPEED TEST: LOCAL VS. NAS"
echo "=========================================="

test_speed() {
    local MOUNT=$1
    local LABEL=$2
    
    echo -e "\n[Testing $LABEL at $MOUNT]"
    
    # 1. WRITE TEST
    echo "Writing 2GB file..."
    # 'oflag=dsync' ensures we bypass the OS write cache
    dd if=/dev/zero of="$MOUNT/$FILE_NAME" bs=$BS count=$COUNT oflag=dsync 2>&1 | grep -E "copied|s,"
    
    # Clear OS buffer cache (requires root)
    echo "Clearing system cache..."
    sync && echo 3 > /proc/sys/vm/drop_caches
    
    # 2. READ TEST
    echo "Reading 2GB file..."
    dd if="$MOUNT/$FILE_NAME" of=/dev/null bs=$BS count=$COUNT 2>&1 | grep -E "copied|s,"
    
    # Cleanup
    rm "$MOUNT/$FILE_NAME"
}


if [ -d "$DRIVE_B" ]; then
    test_speed "$DRIVE_B" "Synology NAS"
else
    echo "Error: $DRIVE_B not mounted."
fi

echo -e "\n=========================================="
echo "Test Complete."