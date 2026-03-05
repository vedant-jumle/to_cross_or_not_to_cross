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
:: Ask upfront: install conda?
:: -----------------------------------------------
where conda >nul 2>&1
if errorlevel 1 (
    echo conda was not found on this system.
    set /p INSTALL_CONDA="Do you want to install Miniconda? [y/n]: "
    if /i "!INSTALL_CONDA!"=="y" (
        goto :install_conda
    ) else (
        echo [ERROR] conda is required. Please install it manually and re-run this script.
        echo Download: https://docs.conda.io/en/latest/miniconda.html
        exit /b 1
    )
) else (
    echo [OK] conda found.
    goto :create_env
)

:: -----------------------------------------------
:: Step 1a: Install Miniconda
:: -----------------------------------------------
:install_conda
echo.
echo [1/5] Installing Miniconda...
curl -o miniconda_installer.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
if errorlevel 1 (
    echo [ERROR] Failed to download Miniconda. Check your internet connection.
    exit /b 1
)
start /wait "" miniconda_installer.exe /S /AddToPath=1 /RegisterPython=1 /D=%USERPROFILE%\Miniconda3
del miniconda_installer.exe
set PATH=%USERPROFILE%\Miniconda3\Scripts;%USERPROFILE%\Miniconda3;%PATH%
call %USERPROFILE%\Miniconda3\Scripts\conda.exe init cmd.exe
call %USERPROFILE%\Miniconda3\Scripts\conda.exe init powershell
echo.
echo [INFO] Miniconda installed successfully.
echo [INFO] Please close this terminal, open a new one, and run this script again.
echo.
pause
exit /b 0

:: -----------------------------------------------
:: Step 2: Create conda environment
:: -----------------------------------------------
:create_env
echo [2/5] Creating conda environment...
call conda activate cv_project >nul 2>&1
if not errorlevel 1 (
    echo [SKIP] Environment 'cv_project' already exists.
    call conda deactivate >nul 2>&1
) else (
    call conda env create -f environment.yml
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
)
echo Done.
echo.

:: -----------------------------------------------
:: Step 3: Clone PIE repo
:: -----------------------------------------------
echo [3/5] Cloning PIE repository...
if exist "_pie_tmp" rmdir /s /q "_pie_tmp"
git clone https://github.com/aras62/PIE _pie_tmp
if errorlevel 1 (
    echo [ERROR] Failed to clone PIE repo. Check your internet connection.
    exit /b 1
)
echo Done.
echo.

:: -----------------------------------------------
:: Step 4: Extract annotations and copy utilities
:: -----------------------------------------------
echo [4/5] Setting up annotations and PIE interface...

if not exist "data\PIE_dataset\clips" mkdir "data\PIE_dataset\clips"
if not exist "src\pie_interface" mkdir "src\pie_interface"

cd _pie_tmp\annotations
tar -xf annotations.zip
tar -xf annotations_attributes.zip
tar -xf annotations_vehicle.zip
cd ..\..

xcopy /e /i /q "_pie_tmp\annotations\annotations" "data\PIE_dataset\annotations\"
xcopy /e /i /q "_pie_tmp\annotations\annotations_attributes" "data\PIE_dataset\annotations_attributes\"
xcopy /e /i /q "_pie_tmp\annotations\annotations_vehicle" "data\PIE_dataset\annotations_vehicle\"

copy "_pie_tmp\utilities\pie_data.py" "src\pie_interface\pie_data.py"
copy "_pie_tmp\utilities\utils.py" "src\pie_interface\utils.py"
copy "_pie_tmp\utilities\data_gen_utils.py" "src\pie_interface\data_gen_utils.py"
if not exist "src\pie_interface\__init__.py" type nul > "src\pie_interface\__init__.py"

echo Done.
echo.

:: -----------------------------------------------
:: Step 5: Cleanup
:: -----------------------------------------------
echo [5/5] Cleaning up temporary files...
rmdir /s /q "_pie_tmp"
echo Done.
echo.

echo ==========================================
echo  Setup complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Restart your terminal (if conda was just installed)
echo   2. Activate the environment:
echo        conda activate cv_project
echo   3. Download video clips:
echo        scripts\windows\download_clips.bat all
echo.
echo ==========================================
