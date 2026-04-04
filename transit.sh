#!/bin/bash

# 1. Define the source and destination base paths
SOURCE_BASE="/mnt/14t_drive"
DEST_BASE="/media/wuchenha/0D9A1B0F0D9A1B0F"

# 2. List your directories here (space-separated)
DIRECTORIES=(
    "QwenASR_mel/fourth_900_videos"
    "QwenASR_mel/fifth_45_videos"
    "QwenASR_mel/sixth_709_videos"
    "QwenASR_mel/seventh_322_videos"
    "QwenASR_mel/eighth_322_videos"
    "QwenASR_mel/ninth_64_videos"
    "OpenSmile/fourth_900_videos"
    "OpenSmile/fifth_45_videos"
    "OpenSmile/sixth_709_videos"
    "OpenSmile/seventh_322_videos"
    "OpenSmile/eighth_322_videos"
    "OpenSmile/ninth_64_videos"
)

echo "Starting batch rsync transfer..."
echo "--------------------------------"

for DIR in "${DIRECTORIES[@]}"; do
    echo "Current Task: Sending $DIR..."
    
    # -a: archive mode
    # -v: verbose
    # -h: human-readable numbers
    # --progress: show progress bar
    # --no-owner --no-group: ignore permission mismatches on external drives
    # --inplace: helps with filesystem stability on external disks
    mkdir -p "$DEST_BASE/$(dirname "$DIR")"
    
    rsync -avh --progress --no-owner --no-group --inplace \
    "$SOURCE_BASE/$DIR/" "$DEST_BASE/$DIR/"

    # Check if the last rsync command succeeded
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $DIR has been transferred."
        echo "--------------------------------"
    else
        echo "ERROR: Transfer of $DIR failed. Stopping script."
        exit 1
    fi
done

echo "All transfers completed successfully!"