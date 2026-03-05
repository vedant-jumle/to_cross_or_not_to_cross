@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo  PIE Dataset - Clip Downloader
echo ==========================================
echo.

:: Check we're in the project root
if not exist "environment.yml" (
    echo [ERROR] Run this script from the project root directory.
    exit /b 1
)

:: Check gdown is available
call conda activate cv_project 2>nul
python -c "import gdown" 2>nul
if errorlevel 1 (
    echo [INFO] Installing gdown...
    pip install -q gdown
)

:: Google Drive file IDs per set
set ID_set01=1sdGWxXL7C_D7dwn5KxZWhLjPHxexPWAQ
set ID_set02=1HLOxiviWIeqRMifAs9K3Pn3yK0zOPhA8
set ID_set03=1WHOOX4rdbJjLD3rFxHW_TWWvNrrS4tWZ
set ID_set04=1WHOOX4rdbJjLD3rFxHW_TWWvNrrS4tWZ
set ID_set05=1qSdH3yfXAIKKspfSo4SWNw3WzYIvckQh
set ID_set06=1-3pn68g82OmOSm66pKAdHNNuMhtJcTKF

:: Target directory
set TARGET=data\PIE_dataset\clips
if not exist "%TARGET%" mkdir "%TARGET%"

:: Parse argument
set ARG=%1

if "%ARG%"=="" goto :ask
if /i "%ARG%"=="all" goto :download_all
goto :download_one

:ask
echo Which sets do you want to download?
echo   1  - set01
echo   2  - set02
echo   3  - set03
echo   4  - set04
echo   5  - set05
echo   6  - set06
echo   all - Download all sets
echo.
set /p ARG="Enter choice: "
if /i "%ARG%"=="all" goto :download_all
goto :download_one

:download_one
set SET=set0%ARG%
call :check_id %SET%
if "!FILE_ID!"=="" (
    echo [ERROR] Invalid choice: %ARG%
    exit /b 1
)
call :download %SET% !FILE_ID!
goto :done

:download_all
for %%S in (set01 set02 set03 set04 set05 set06) do (
    call :check_id %%S
    call :download %%S !FILE_ID!
)
goto :done

:: -----------------------------------------------
:: Subroutine: check_id <setname>
:: Sets FILE_ID variable
:: -----------------------------------------------
:check_id
set FILE_ID=!ID_%1!
exit /b 0

:: -----------------------------------------------
:: Subroutine: download <setname> <file_id>
:: -----------------------------------------------
:download
set SETNAME=%1
set FID=%2
set OUTFILE=%TARGET%\%SETNAME%.zip

if exist "%OUTFILE%" (
    echo [SKIP] %OUTFILE% already exists. Delete it to re-download.
    exit /b 0
)

echo [DOWNLOADING] %SETNAME% ^(ID: %FID%^)...
python -m gdown --id %FID% -O %OUTFILE%
if errorlevel 1 (
    echo [ERROR] Failed to download %SETNAME%.
    exit /b 1
)

echo [EXTRACTING] %SETNAME%...
tar -xf %OUTFILE% -C %TARGET%
echo [DONE] %SETNAME%
echo.
exit /b 0

:done
echo ==========================================
echo  Download complete!
echo ==========================================
