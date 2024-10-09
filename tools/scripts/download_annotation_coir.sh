#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory
mkdir -p annotation/cc-coir

# Download Train annotations
echo -e "Downloading ${BLUE}CC-CoIR Train${NC} annotations..."
wget -q --show-progress "https://huggingface.co/lucas-ventura/CoVR/resolve/main/webvid-covr.ckpt" -P annotation/cc-coir/



