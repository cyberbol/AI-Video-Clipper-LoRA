@echo off
TITLE AI Clipper Installer v3.2 (Conflict Fix)
color 0B

echo ======================================================================
echo       AI VIDEO CLIPPER ^& LORA CAPTIONER - INSTALLER
echo             Optimized for Stability ^& RTX 5090 Support
echo ======================================================================
echo.

:MENU
echo ======================================================================
echo    SELECT YOUR GRAPHICS CARD (GPU)
echo ======================================================================
echo.
echo  [1] STANDARD (RTX 4090, 3090, 20xx)
echo      - Installs PyTorch 2.8.0 (CUDA 12.6)
echo      - Best for compatibility with WhisperX.
echo.
echo  [2] NEXT-GEN (RTX 5090 / Blackwell)
echo      - Installs PyTorch Nightly (CUDA 13.0)
echo      - REQUIRED for 50-series cards.
echo.
set /p gpu_choice="Type 1 or 2 and press ENTER: "

if "%gpu_choice%"=="1" goto SETUP_STABLE
if "%gpu_choice%"=="2" goto SETUP_EXPERIMENTAL
goto MENU

:SETUP_STABLE
cls
color 0A
echo [INFO] Selected: STANDARD MODE (Stable / CUDA 12.6)
:: ZMIANA: Podbilem wersje do 2.8.0, zeby WhisperX nie robil awantury
set "TORCH_CMD=pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
goto START_INSTALL

:SETUP_EXPERIMENTAL
cls
color 0D
echo [WARNING] Selected: RTX 5090 MODE (Nightly / CUDA 13.0)
:: Tutaj nadal Nightly - moze wystapic czerwony tekst na koncu, ale to normalne dla wersji beta
set "TORCH_CMD=pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130"
goto START_INSTALL

:START_INSTALL
echo.
echo [STEP 1/6] Creating isolated environment (venv)...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [STEP 2/6] Installing FFmpeg...
:: Instalujemy ffmpeg-python explicit, zeby uniknac problemow z audio backend
pip install ffmpeg-python

echo.
echo [STEP 3/6] Installing GPU Engine...
python -m pip install --upgrade pip
%TORCH_CMD%

echo.
echo [STEP 4/6] Installing WhisperX (Clean Mode)...
:: Instalujemy WhisperX bez wymuszania zmiany wersji Torcha (no-deps dla torcha), 
:: ale pozwalamy mu dobrac inne biblioteki.
pip install git+https://github.com/m-bain/whisperX.git

echo.
echo [STEP 5/6] Installing Vision AI & App Components...
:: Instalujemy reszte paczek
pip install qwen-vl-utils accelerate transformers streamlit moviepy "pillow<11.0" tk psutil

echo.
echo [STEP 6/6] Final GPU Engine Verify...
:: Upewniamy sie, ze wersja Torcha jest ta, ktora wybralismy w MENU
%TORCH_CMD%

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo You can now run the program using "3. Run.bat".
echo.
pause