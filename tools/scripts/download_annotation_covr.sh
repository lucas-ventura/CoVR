#!/bin/bash

# Define colors
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory
mkdir -p annotation/webvid-covr

# Download Train annotations
echo -e "Downloading ${BLUE}WebVid-CoVR Train${NC} annotations..."
wget http://imagine.enpc.fr/~ventural/covr/dataset/webvid2m-covr_train.csv -q -P annotation/webvid-covr/

# Download Val annotations
echo -e "Downloading ${BLUE}WebVid-CoVR Val${NC} annotations..."
wget http://imagine.enpc.fr/~ventural/covr/dataset/webvid8m-covr_val.csv -q -P annotation/webvid-covr/

# Download Test annotations
echo -e "Downloading ${BLUE}WebVid-CoVR Test${NC} annotations..."
wget http://imagine.enpc.fr/~ventural/covr/dataset/webvid8m-covr_test.csv -q -P annotation/webvid-covr/
