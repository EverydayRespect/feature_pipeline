#!/bin/bash

sudo apt update
sudo apt install -y cifs-utils docker.io curl
sudo systemctl start docker
sudo systemctl enable docker

# Step 1: 创建挂载点目录（可选）
sudo mkdir -p /mnt/synology

# Step 2: 将 NAS 挂载到本地（使用 CIFS 协议）
# 替换密码为实际密码，推荐写成凭据文件来避免明文密码暴露
sudo mount -t cifs //169.254.9.31/data /mnt/synology \
    -o username=wuchenha,password=xCCp0DXU,vers=3.0

# Step 3: 进入挂载目录
cd /mnt/synology

# Step 4: 下载 Milvus 脚本
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Step 5: 运行 Milvus（默认端口：19530）
bash standalone_embed.sh start