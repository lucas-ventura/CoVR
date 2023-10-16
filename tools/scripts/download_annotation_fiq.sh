#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory
mkdir -p annotation/fashion-iq

# Download Train annotations
echo -e "Downloading ${BLUE}fashion-iq Train${NC} annotations..."
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.dress.train.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.shirt.train.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.toptee.train.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.dress.train.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.shirt.train.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.toptee.train.json -q -P annotation/fashion-iq/

# Download Val annotations
echo -e "Downloading ${BLUE}fashion-iq Val${NC} annotations..."
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.dress.val.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.shirt.val.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.toptee.val.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.dress.val.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.shirt.val.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.toptee.val.json -q -P annotation/fashion-iq/

# Download Test annotations
echo -e "Downloading ${BLUE}fashion-iq Test${NC} annotations..."
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.dress.test.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.shirt.test.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions/cap.toptee.test.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.dress.test.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.shirt.test.json -q -P annotation/fashion-iq/
wget https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits/split.toptee.test.json -q -P annotation/fashion-iq/
