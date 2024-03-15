#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory
mkdir -p annotation/fashion-iq

download_file() {
    local url=$1
    local path=$2
    local file=$(basename "$url")
    if [ ! -f "$path/$file" ]; then
        echo -e "Downloading ${BLUE}$file${NC}..."
        wget $url -q -P $path
    else
        echo -e "${BLUE}$file${NC} already exists, skipping download."
    fi
}

# Download Train annotations
echo "Checking Fashion-IQ Train annotations..."
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.dress.train.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.shirt.train.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.toptee.train.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.dress.train.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.shirt.train.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.toptee.train.json" "annotation/fashion-iq"

# Download Val annotations
echo "Checking Fashion-IQ Val annotations..."
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.dress.val.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.shirt.val.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.toptee.val.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.dress.val.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.shirt.val.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.toptee.val.json" "annotation/fashion-iq"

# Download Test annotations
echo "Checking Fashion-IQ Test annotations..."
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.dress.test.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.shirt.test.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.toptee.test.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.dress.test.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.shirt.test.json" "annotation/fashion-iq"
download_file "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.toptee.test.json" "annotation/fashion-iq"

python tools/scripts/merge_fiq_annotations.py