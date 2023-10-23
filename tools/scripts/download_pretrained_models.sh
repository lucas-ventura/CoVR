#!/bin/bash

# Define colors
BLUE='\033[0;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# webvid-covr
url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/webvid-covr.ckpt"
dir="outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/"
filename="ckpt_4.ckpt"

# Create the directory if it doesn't exist
mkdir -p "$dir"

# Check if the file already exists
if [ -e "$dir/$filename" ]; then
    echo -e "The ${BLUE}WebVid-CoVR${NC} checkpoint already exists."
else
    # Download the file and check if it's downloaded correctly
    echo -e "Downloading ${BLUE}WebVid-CoVR${NC} checkpoint..."
    if wget -q --show-progress "$url" -O "$dir/$filename"; then
        echo -e "${GREEN}Download successful.${NC}"
    else
        echo -e "${RED}Download failed.${NC}"
    fi
fi


# webvid-covr fine-tuned on CIRR
url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/cirr_ft-covr%2Bgt.ckpt"
dir="outputs/cirr/blip-large/webvid-covr/tv-False_loss-hnnce_lr-0.0001/base/"
filename="ckpt_5.ckpt"

# Create the directory if it doesn't exist
mkdir -p "$dir"

# Check if the file already exists
if [ -e "$dir/$filename" ]; then
    echo -e "The ${BLUE}WebVid-CoVR CIRR finetuned${NC} checkpoint already exists."
else
    # Download the file and check if it's downloaded correctly
    echo -e "Downloading ${BLUE}WebVid-CoVR CIRR${NC} finetuned checkpoint..."
    if wget -q --show-progress "$url" -O "$dir/$filename"; then
        echo -e "${GREEN}Download successful.${NC}"
    else
        echo -e "${RED}Download failed.${NC}"
    fi
fi
