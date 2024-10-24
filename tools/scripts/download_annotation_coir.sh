#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory
mkdir -p annotation/cc-coir

# Download Train annotations
echo -e "Downloading ${BLUE}CC-CoIR Train${NC} annotations..."
wget -q --show-progress https://huggingface.co/datasets/lucas-ventura/CC-CoIR/resolve/main/cc-coir_train.csv -P annotation/cc-coir/