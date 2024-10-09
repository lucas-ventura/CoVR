#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [covr|coir|cirr|fiq|all]"
    exit 1
fi

case "$1" in
    covr)
        bash tools/scripts/download_annotation_covr.sh
        ;;
    coir)
        bash tools/scripts/download_annotation_coir.sh
        ;;
    cirr)
        bash tools/scripts/download_annotation_cirr.sh
        ;;
    fiq)
        bash tools/scripts/download_annotation_fiq.sh
        ;;
    fashion-iq)
        bash tools/scripts/download_annotation_fiq.sh
        ;;
    fashioniq)
        bash tools/scripts/download_annotation_fiq.sh
        ;;
    all)
        bash tools/scripts/download_annotation_covr.sh
        bash tools/scripts/download_annotation_cirr.sh
        bash tools/scripts/download_annotation_fiq.sh
        ;;
    *)
        echo "Invalid argument. Usage: $0 [covr|coir|cirr|fiq|all]"
        exit 1
        ;;
esac
