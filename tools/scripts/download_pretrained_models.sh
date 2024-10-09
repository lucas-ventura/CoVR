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
choose_model() {
    echo "Select the model to download:"
    echo -e "1) ${BLUE}BLIP 1 (CoVR)${NC}"
    echo -e "2) ${BLUE}BLIP 2 (CoVR-2)${NC}"
    echo -e "Press Enter for default (${GREEN}BLIP 2${NC})"
    read -p "Enter your choice (1/2): " model_choice
    case $model_choice in
        1) choose_download_option_blip1 ;;
        *) choose_download_option_blip2 ;;
    esac
}

### BLIP 1 (CoVR) ###
choose_download_option_blip1() {
    echo ""
    echo "Select the BLIP 1 model finetuned with dataset:"
    echo -e "1) ${BLUE}All${NC}"
    echo -e "2) ${BLUE}WebVid-CoVR${NC}"
    echo -e "3) ${BLUE}CIRR (WebVid-CoVR)${NC}"
    echo -e "4) ${BLUE}FashionIQ (WebVid-CoVR)${NC}"
    echo -e "Press Enter for default (${GREEN}All${NC})"
    read -p "Enter your choice (1/2/3/4): " choice
    case $choice in
        2) download1_webvid_covr ;;
        3) download1_cirr ;;
        4) download1_fiq ;;
        *) download1_all ;;
    esac
}

# WebVid-CoVR
download1_webvid_covr() {
    url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/webvid-covr.ckpt"
    dir="outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/"
    filename="ckpt_4.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

# WebVid-CoVR + finetuned on CIRR
download1_cirr() {
    url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/cirr_ft-covr%2Bgt.ckpt"
    dir="outputs/cirr/blip-large/webvid-covr/tv-False_loss-hnnce_lr-0.0001/base/"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

# WebVid-CoVR + finetuned on FashionIQ
download1_fiq() {
    url="https://huggingface.co/lucas-ventura/CoVR/resolve/main/fashioniq-all-ft_covr.ckpt"
    dir="outputs/fashioniq-all/blip-large/webvid-covr/tv-False_loss-hnnce_lr-0.0001/base"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

# all models
download1_all() {
    download1_webvid_covr
    download1_cirr
    download1_fiq
}

### BLIP 2 (CoVR-2) ###
choose_download_option_blip2() {
    echo ""
    echo "Select the BLIP 2 model finetuned with dataset:"
    echo -e "1) ${BLUE}All${NC}"
    echo -e "2) ${BLUE}WebVid-CoVR + CC-CoIR${NC}"
    echo -e "3) ${BLUE}CC-CoIR${NC}"
    echo -e "4) ${BLUE}WebVid-CoVR${NC}"
    echo -e "5) ${BLUE}CIRR (WebVid-CoVR + CC-CoIR)${NC}"
    echo -e "6) ${BLUE}CIRR (CC-CoIR)${NC}"
    echo -e "7) ${BLUE}FashionIQ (WebVid-CoVR + CC-CoIR)${NC}"
    echo -e "8) ${BLUE}FashionIQ (CC-CoIR)${NC}"
    echo -e "Press Enter for default (${GREEN}All${NC})"
    read -p "Enter your choice (1/2/3/4/5/6/7/8): " choice
    case $choice in
        2) download2_covr_coir ;;
        3) download2_coir ;;
        4) download2_covr ;;
        5) download2_cirr_covr_coir ;;
        6) download2_cirr_coir ;;
        7) download2_fiq_covr_coir ;;
        8) download2_fiq_coir ;;
        *) download2_all ;;
    esac
}

download2_covr_coir() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/coir%2Bcovr.ckpt"
    dir="outputs/cc-coir+webvid-covr/blip2-coco/blip2-l-coco/tv-False_loss-hnnce_lr-1e-05/base/"
    filename="ckpt_0.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

download2_coir() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/coir.ckpt"
    dir="outputs/cc-coir/blip2-coco/blip2-l-coco/tv-False_loss-hnnce_lr-2e-05/base/"
    filename="ckpt_2.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

download2_covr() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/webvid-covr.ckpt"
    dir="outputs/webvid-covr/blip2-coco/blip2-l-coco/tv-False_loss-hnnce_lr-2e-05/base/"
    filename="ckpt_4.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

download2_cirr_covr_coir() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/cirr_ft-coir%2Bcovr.ckpt"
    dir="outputs/cirr/blip2-coco/blip2-l-coco_coir+covr/tv-False_loss-hnnce_lr-0.0001/base/"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

download2_cirr_coir() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/cirr_ft-coir.ckpt"
    dir="outputs/cirr/blip2-coco/blip2-l-coco_coir/tv-False_loss-hnnce_lr-0.0001/base/"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

download2_fiq_covr_coir() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/fashioniq-all-ft_coir%2Bcovr.ckpt"
    dir="outputs/fashioniq-all/blip2-coco/blip2-l-coco_coir+covr/tv-False_loss-hnnce_lr-0.0001/base/"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}
download2_fiq_coir() {
    url="https://huggingface.co/lucas-ventura/CoVR2/resolve/main/fashioniq-all-ft_coir.ckpt"
    dir="outputs/fashioniq-all/blip2-coco/blip2-l-coco_coir/tv-False_loss-hnnce_lr-0.0001/base/"
    filename="ckpt_5.ckpt"
    download_checkpoint "$url" "$dir" "$filename"
}

download2_all() {
    download2_covr_coir
    download2_coir
    download2_covr
    download2_cirr_covr_coir
    download2_cirr_coir
    download2_fiq_covr_coir
    download2_fiq_coir
}

# Main script execution
choose_model