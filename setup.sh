#!/bin/bash

sudo apt update
sudo apt install -y cifs-utils docker.io curl
sudo systemctl start docker
sudo systemctl enable docker

# Step 1: create a mount point
sudo mkdir -p /mnt/synology

# Step 2: mount NAS to local mount point
sudo mount -t cifs //169.254.9.31/home /mnt/synology -o username=wuchenha,password='xCCp0DXU',vers=3.0,uid=1008,gid=1010,file_mode=0777,dir_mode=0777

# Step 3: change pwd to mount point
cd /mnt/synology

# Step 4: download the script for Milvus
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Step 5: run Milvus (default port: 19530)
bash standalone_embed.sh start