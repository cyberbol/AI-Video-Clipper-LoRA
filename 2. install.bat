@echo off
TITLE AI Clipper Installer - Universal GPU Edition
color 0B

echo ======================================================================
echo          AI VIDEO CLIPPER ^& LORA CAPTIONER - INSTALLER
echo                       Created by Cyberbol
echo ======================================================================
echo.
echo [IMPORTANT] Before we start, make sure you have:
echo 1. Python 3.10 installed (Add to PATH checked)
echo 2. Visual Studio Build Tools (C++ Desktop Development) installed.
echo 3. Git installed.
echo 4. Latest NVIDIA Drivers.
echo.

:MENU
echo ======================================================================
echo    SELECT YOUR GRAPHICS CARD (GPU)
echo ======================================================================
echo.
echo  [1] STANDARD (RTX 4090, 3090, 2080, etc.)
echo      - Uses Stable PyTorch (CUDA 11.8 / 12.1).
echo      - Recommended for 99%% of users.
echo.
echo  [2] EXPERIMENTAL (RTX 5090 / Blackwell)
echo      - Uses Nightly PyTorch (CUDA 12.8+).
echo      - WARNING: This is experimental! It might not work yet.
echo      - Use only if Option [1] fails to detect your GPU.
echo.
set /p gpu_choice="Type 1 or 2 and press ENTER: "

if "%gpu_choice%"=="1" goto SETUP_STABLE
if "%gpu_choice%"=="2" goto SETUP_EXPERIMENTAL
echo Invalid choice. Please try again.
goto MENU

:SETUP_STABLE
cls
color 0A
echo [INFO] Selected: STANDARD MODE (Stable)
set "TORCH_CMD=pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118"
goto START_INSTALL

:SETUP_EXPERIMENTAL
cls
color 0D
echo [WARNING] Selected: RTX 5090 MODE (Experimental/Nightly)
echo.
echo **************************************************************
echo  DISCLAIMER:
echo  You are installing a 'Nightly' (Beta) version of PyTorch
echo  required for the RTX 5090. This build changes daily and
echo  is NOT guaranteed to be stable. Use at your own risk.
echo **************************************************************
echo.
pause
set "TORCH_CMD=pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
goto START_INSTALL

:START_INSTALL
echo.
echo [STEP 1/6] Creating isolated environment (venv)...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [STEP 2/6] Installing FFmpeg...
winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements

echo.
echo [STEP 3/6] Installing GPU Engines (Selected Version)...
echo Command: %TORCH_CMD%
%TORCH_CMD%

echo.
echo [STEP 4/6] Installing WhisperX (This will break Torch temporarily)...
echo Please wait, compiling C++ components...
pip install git+https://github.com/m-bain/whisperX.git

echo.
echo [STEP 5/6] AUTO-REPAIR: Fixing GPU Compatibility...
echo WhisperX installed a CPU version. We are removing it now...
pip uninstall torch torchvision torchaudio -y
echo.
echo Re-installing the correct GPU version for your selection...
%TORCH_CMD%

echo.
echo [STEP 6/6] Installing Vision AI and Interface...
pip install qwen-vl-utils accelerate transformers streamlit moviepy "pillow<11.0"

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo You can now run the program using "3. Run.bat".
echo.
pause