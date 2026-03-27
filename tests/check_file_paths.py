import os
from pathlib import Path

base_dir = Path("/mnt/synology/BWVs")
subdirs = ["eighth_322_videos","fifth_45_videos","fourth_900_videos","ninth_64_videos","seventh_322_videos","sixth_709_videos"]

for subdir in subdirs:
    subdir = base_dir / subdir
    mp4_files = list(subdir.rglob("*.mp4"))
    for mp4_file in mp4_files:
        rel_path = mp4_file.relative_to(base_dir)
        print(rel_path)