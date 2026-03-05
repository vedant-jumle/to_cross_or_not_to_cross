#!/bin/bash
set -e

echo "=========================================="
echo " to_cross_or_not_to_cross - Setup Script"
echo "=========================================="
echo

# Check we're in the project root
if [ ! -f "environment.yml" ]; then
    echo "[ERROR] Run this script from the project root directory."
    exit 1
fi

# -----------------------------------------------
# Ask upfront: install conda?
# -----------------------------------------------
if ! command -v conda &>/dev/null; then
    echo "conda was not found on this system."
    read -rp "Do you want to install Miniconda? [y/n]: " INSTALL_CONDA
    if [[ "$INSTALL_CONDA" =~ ^[Yy]$ ]]; then
        echo
        echo "[1/5] Installing Miniconda..."
        curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda_installer.sh
        bash miniconda_installer.sh -b -p "$HOME/miniconda3"
        rm miniconda_installer.sh
        export PATH="$HOME/miniconda3/bin:$PATH"
        conda init bash
        echo "[INFO] Miniconda installed. Restart your terminal after setup, or run: source ~/.bashrc"
        echo "Done."
        echo
    else
        echo "[ERROR] conda is required. Please install it manually and re-run this script."
        echo "Download: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
else
    echo "[OK] conda found."
    echo
fi

# -----------------------------------------------
# Step 2: Create conda environment
# -----------------------------------------------
echo "[2/5] Creating conda environment..."
if conda env list | grep -q "cv_project"; then
    echo "[SKIP] Environment 'cv_project' already exists."
else
    conda env create -f environment.yml
fi
echo "Done."
echo

# -----------------------------------------------
# Step 3: Clone PIE repo
# -----------------------------------------------
echo "[3/5] Cloning PIE repository..."
rm -rf _pie_tmp
git clone https://github.com/aras62/PIE _pie_tmp
echo "Done."
echo

# -----------------------------------------------
# Step 4: Extract annotations and copy utilities
# -----------------------------------------------
echo "[4/5] Setting up annotations and PIE interface..."

mkdir -p data/PIE_dataset/clips
mkdir -p src/pie_interface

cd _pie_tmp/annotations
unzip -q annotations.zip
unzip -q annotations_attributes.zip
unzip -q annotations_vehicle.zip
cd ../..

cp -r _pie_tmp/annotations/annotations            data/PIE_dataset/annotations
cp -r _pie_tmp/annotations/annotations_attributes data/PIE_dataset/annotations_attributes
cp -r _pie_tmp/annotations/annotations_vehicle    data/PIE_dataset/annotations_vehicle

cp _pie_tmp/utilities/pie_data.py       src/pie_interface/pie_data.py
cp _pie_tmp/utilities/utils.py          src/pie_interface/utils.py
cp _pie_tmp/utilities/data_gen_utils.py src/pie_interface/data_gen_utils.py
touch src/pie_interface/__init__.py

echo "Done."
echo

# -----------------------------------------------
# Step 5: Cleanup
# -----------------------------------------------
echo "[5/5] Cleaning up temporary files..."
rm -rf _pie_tmp
echo "Done."
echo

echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. If conda was just installed, restart your terminal or run:"
echo "       source ~/.bashrc"
echo "  2. Activate the environment:"
echo "       conda activate cv_project"
echo "  3. Download video clips:"
echo "       bash scripts/linux/download_clips.sh all"
echo
echo "=========================================="
