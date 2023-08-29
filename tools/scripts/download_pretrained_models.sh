#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Download WebVid-CoVR checkpoint
dir=outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/
mkdir -p $dir
echo -e "${BLUE}Downloading WebVid-CoVR checkpoint...${NC}"
wget "https://drive.google.com/uc?id=18agaaO2RNJP1W86Kls5z6bxp8_B5sVtP&export=download&confirm=t&uuid=2d190ab0-c0d9-4092-9c29-d20a49fffbd9" -q --show-progress -O $dir/ckpt_4.ckpt

