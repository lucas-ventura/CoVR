#!/bin/bash

# Define colors
BLUE='\033[0;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to download a checkpoint
download_checkpoint() {
    local url="$1"
    local dir="$2"
    local filename="$3"
    local overwrite="$4" # Added a variable for the "overwrite" option

    # Create the directory if it doesn't exist
    mkdir -p "$dir"

    # Check if the file already exists and the "overwrite" option is not provided
    if [ -e "$dir/$filename" ] && [ "$overwrite" != "overwrite" ]; then
        echo -e "The ${BLUE}$filename${NC} checkpoint already exists in $dir."
    else
        # Download the file and check if it's downloaded correctly
        echo -e "Downloading ${BLUE}$filename${NC} checkpoint..."
        if wget -q --show-progress "$url" -O "$dir/$filename"; then
            echo -e "${GREEN}Download successful.${NC}"
        else
            echo -e "${RED}Download failed.${NC}"
        fi
    fi
}

# webvid-covr
url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/webvid-covr.ckpt"
dir="outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/"
filename="ckpt_4.ckpt"

# Call the download_checkpoint function with the "overwrite" argument
download_checkpoint "$url" "$dir" "$filename" "$1"

# webvid-covr fine-tuned on CIRR
url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/cirr_ft-covr%2Bgt.ckpt"
dir="outputs/cirr/blip-large/webvid-covr/tv-False_loss-hnnce_lr-0.0001/base/"
filename="ckpt_5.ckpt"

# Call the download_checkpoint function with the "overwrite" argument
download_checkpoint "$url" "$dir" "$filename" "$1"
