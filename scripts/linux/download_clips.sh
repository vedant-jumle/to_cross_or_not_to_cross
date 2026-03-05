#!/bin/bash
set -e

echo "=========================================="
echo " PIE Dataset - Clip Downloader"
echo "=========================================="
echo

# Check we're in the project root
if [ ! -f "environment.yml" ]; then
    echo "[ERROR] Run this script from the project root directory."
    exit 1
fi

# Check gdown is available
if ! python -c "import gdown" 2>/dev/null; then
    echo "[INFO] Installing gdown..."
    pip install -q gdown
fi

# Google Drive file IDs per set
declare -A FILE_IDS
FILE_IDS[set01]="1sdGWxXL7C_D7dwn5KxZWhLjPHxexPWAQ"
FILE_IDS[set02]="1HLOxiviWIeqRMifAs9K3Pn3yK0zOPhA8"
FILE_IDS[set03]="1WHOOX4rdbJjLD3rFxHW_TWWvNrrS4tWZ"
FILE_IDS[set04]="1WHOOX4rdbJjLD3rFxHW_TWWvNrrS4tWZ"
FILE_IDS[set05]="1qSdH3yfXAIKKspfSo4SWNw3WzYIvckQh"
FILE_IDS[set06]="1-3pn68g82OmOSm66pKAdHNNuMhtJcTKF"

TARGET="data/PIE_dataset/clips"
mkdir -p "$TARGET"

download_set() {
    local SETNAME=$1
    local FID=${FILE_IDS[$SETNAME]}
    local OUTFILE="$TARGET/$SETNAME.zip"

    if [ -z "$FID" ]; then
        echo "[ERROR] Unknown set: $SETNAME"
        exit 1
    fi

    if [ -f "$OUTFILE" ]; then
        echo "[SKIP] $OUTFILE already exists. Delete it to re-download."
        return
    fi

    echo "[DOWNLOADING] $SETNAME (ID: $FID)..."
    python -m gdown --id "$FID" -O "$OUTFILE"

    echo "[EXTRACTING] $SETNAME..."
    unzip -q "$OUTFILE" -d "$TARGET"
    echo "[DONE] $SETNAME"
    echo
}

ARG=${1:-""}

if [ -z "$ARG" ]; then
    echo "Which set do you want to download?"
    echo "  1  - set01"
    echo "  2  - set02"
    echo "  3  - set03"
    echo "  4  - set04"
    echo "  5  - set05"
    echo "  6  - set06"
    echo "  all - Download all sets"
    echo
    read -rp "Enter choice: " ARG
fi

if [ "$ARG" = "all" ]; then
    for SET in set01 set02 set03 set04 set05 set06; do
        download_set "$SET"
    done
elif [[ "$ARG" =~ ^[1-6]$ ]]; then
    download_set "set0$ARG"
else
    echo "[ERROR] Invalid choice: $ARG"
    exit 1
fi

echo "=========================================="
echo " Download complete!"
echo "=========================================="
