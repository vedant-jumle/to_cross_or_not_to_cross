@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo  to_cross_or_not_to_cross - Setup Script
echo ==========================================
echo.

:: Check we're in the project root
if not exist "environment.yml" (
    echo [ERROR] Run this script from the project root directory.
    exit /b 1
)

:: -----------------------------------------------
:: Step 1: Conda environment
:: -----------------------------------------------
echo [1/4] Creating conda environment...
call conda env create -f environment.yml
if errorlevel 1 (
    echo [WARN] Environment creation failed - it may already exist. Skipping.
)
echo Done.
echo.

:: -----------------------------------------------
:: Step 2: Clone PIE repo
:: -----------------------------------------------
echo [2/4] Cloning PIE repository...
if exist "_pie_tmp" rmdir /s /q "_pie_tmp"
git clone https://github.com/aras62/PIE _pie_tmp
if errorlevel 1 (
    echo [ERROR] Failed to clone PIE repo. Check your internet connection.
    exit /b 1
)
echo Done.
echo.

:: -----------------------------------------------
:: Step 3: Extract annotations and copy utilities
:: -----------------------------------------------
echo [3/4] Setting up annotations and PIE interface...

:: Create target dirs
if not exist "data\PIE_dataset\clips" mkdir "data\PIE_dataset\clips"
if not exist "src\pie_interface" mkdir "src\pie_interface"

:: Extract annotation zips
cd _pie_tmp\annotations
tar -xf annotations.zip
tar -xf annotations_attributes.zip
tar -xf annotations_vehicle.zip
cd ..\..

:: Copy extracted annotations
xcopy /e /i /q "_pie_tmp\annotations\annotations" "data\PIE_dataset\annotations\"
xcopy /e /i /q "_pie_tmp\annotations\annotations_attributes" "data\PIE_dataset\annotations_attributes\"
xcopy /e /i /q "_pie_tmp\annotations\annotations_vehicle" "data\PIE_dataset\annotations_vehicle\"

:: Copy PIE interface utilities
copy "_pie_tmp\utilities\pie_data.py" "src\pie_interface\pie_data.py"
copy "_pie_tmp\utilities\utils.py" "src\pie_interface\utils.py"
copy "_pie_tmp\utilities\data_gen_utils.py" "src\pie_interface\data_gen_utils.py"

:: Create __init__.py if missing
if not exist "src\pie_interface\__init__.py" type nul > "src\pie_interface\__init__.py"

echo Done.
echo.

:: -----------------------------------------------
:: Step 4: Cleanup
:: -----------------------------------------------
echo [4/4] Cleaning up temporary files...
rmdir /s /q "_pie_tmp"
echo Done.
echo.

:: -----------------------------------------------
:: Final instructions
:: -----------------------------------------------
echo ==========================================
echo  Setup complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Activate the environment:
echo        conda activate cv_project
echo.
echo   2. Place PIE dataset zips in:
echo        data\PIE_dataset\clips\
echo      (set01.zip, set02.zip, ... set06.zip)
echo.
echo   3. Extract each zip:
echo        cd data\PIE_dataset\clips
echo        tar -xf set01.zip
echo        tar -xf set02.zip
echo        ... etc
echo.
echo ==========================================
