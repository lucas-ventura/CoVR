#!/bin/bash

# Define colors
BLUE='\033[0;34m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

download_checkpoint() {
    local url="$1"
    local dir="$2"
    local filename="$3"

    # Create the directory if it doesn't exist
    mkdir -p "$dir"

    # Check if the file already exists
    if [ -e "$dir/$filename" ]; then
        echo -e "The ${BLUE}$filename${NC} checkpoint already exists in $dir."
        # Ask the user if they want to overwrite the file, defaulting to 'no'
        read -p "Do you want to overwrite? [y/N]: " overwrite
        case $overwrite in
            [Yy]* ) ;;
            * ) return ;;
        esac
    fi

    # Download the file and check if it's downloaded correctly
    echo -e "Downloading ${BLUE}$filename${NC} checkpoint..."
    if wget -q --show-progress "$url" -O "$dir/$filename"; then
        echo -e "${GREEN}Download successful.${NC}"
    else
        echo -e "${RED}Download failed.${NC}"
    fi
}

# Function to display download options
choose_download_option() {
    echo "Select the model to download:"
    echo "1) All"
    echo "2) WebVid-CoVR"
    echo "3) CIRR"
    echo "4) FashionIQ"
    echo "Press Enter for default (All)"
    read -p "Enter your choice (1/2/3/4): " choice
    case $choice in
        2) download_webvid_covr ;;
        3) download_cirr ;;
        4) download_fiq ;;
        *) download_all ;;
    esac
}

# WebVid-CoVR
download_webvid_covr() {
    url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/webvid-covr.ckpt"
    dir="outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/"
    filename="ckpt_4.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

# WebVid-CoVR + finetuned on CIRR
download_cirr() {
    url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/cirr_ft-covr%2Bgt.ckpt"
    dir="outputs/cirr/blip-large/webvid-covr/tv-False_loss-hnnce_lr-0.0001/base/"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

# WebVid-CoVR + finetuned on FashionIQ
download_fiq() {
    url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/fashioniq-all-ft_covr.ckpt"
    dir="outputs/fashioniq-all/blip-large/webvid-covr/tv-False_loss-hnnce_lr-0.0001/base"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

# all models
download_all() {
    download_webvid_covr
    download_cirr
    download_fiq
}

# Main script execution
choose_download_option