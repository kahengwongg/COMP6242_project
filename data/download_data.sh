#!/bin/bash
# Anime Face Dataset Download Script
# Dataset source: https://github.com/bchao1/Anime-Face-Dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/anime_faces"

echo "========================================"
echo "Anime Face Dataset Download Script"
echo "========================================"

# Check if data already exists
if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "Dataset already exists at: $DATA_DIR"
    echo "To re-download, please delete the directory first"
    exit 0
fi

# Create directory
mkdir -p "$DATA_DIR"

# Download method 1: Use Kaggle dataset (requires Kaggle API configuration)
# Using fallback method: download from GitHub repository

echo "Downloading dataset..."
echo "Note: If download fails, please manually download the dataset to data/anime_faces/ directory"

# Try downloading with wget or curl
if command -v wget &> /dev/null; then
    # Download from fallback source
    echo "Downloading with wget..."
    wget -q --show-progress -O "${SCRIPT_DIR}/anime_faces.zip" \
        "https://github.com/bchao1/Anime-Face-Dataset/archive/refs/heads/master.zip" || {
        echo "wget download failed, please manually download the dataset"
        echo "Download URL: https://www.kaggle.com/datasets/soumikrakshit/anime-faces"
        exit 1
    }
elif command -v curl &> /dev/null; then
    echo "Downloading with curl..."
    curl -L -o "${SCRIPT_DIR}/anime_faces.zip" \
        "https://github.com/bchao1/Anime-Face-Dataset/archive/refs/heads/master.zip" || {
        echo "curl download failed, please manually download the dataset"
        echo "Download URL: https://www.kaggle.com/datasets/soumikrakshit/anime-faces"
        exit 1
    }
else
    echo "Error: wget or curl is required to download the dataset"
    echo "Please manually download the dataset to data/anime_faces/ directory"
    echo "Download URL: https://www.kaggle.com/datasets/soumikrakshit/anime-faces"
    exit 1
fi

# Extract
echo "Extracting..."
if command -v unzip &> /dev/null; then
    unzip -q "${SCRIPT_DIR}/anime_faces.zip" -d "${SCRIPT_DIR}/"
    # Move files to the correct location
    if [ -d "${SCRIPT_DIR}/Anime-Face-Dataset-master/data" ]; then
        rm -rf "$DATA_DIR"
        mv "${SCRIPT_DIR}/Anime-Face-Dataset-master/data" "$DATA_DIR"
        rm -rf "${SCRIPT_DIR}/Anime-Face-Dataset-master"
    fi
    rm "${SCRIPT_DIR}/anime_faces.zip"
else
    echo "Warning: unzip not found, please manually extract ${SCRIPT_DIR}/anime_faces.zip"
    exit 1
fi

# Count number of files
FILE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l | tr -d ' ')

echo "========================================"
echo "Download complete!"
echo "Data directory: $DATA_DIR"
echo "Image count: $FILE_COUNT"
echo "========================================"

# Warn if too few images were downloaded
if [ "$FILE_COUNT" -lt 10000 ]; then
    echo ""
    echo "Warning: Too few images downloaded, consider manually downloading the full dataset"
    echo "The full dataset contains approximately 63,000 images"
    echo "Download URL: https://www.kaggle.com/datasets/soumikrakshit/anime-faces"
fi
