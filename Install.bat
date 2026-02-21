@echo off
setlocal EnableDelayedExpansion
:: AI Video Clipper & LoRA Captioner - Windows Installer (v5.0 Staging)

TITLE AI Clipper Installer - UV Edition
color 0B

:: UV Optimizations
set UV_HTTP_TIMEOUT=3600
set UV_LINK_MODE=hardlink
set UV_CACHE_DIR=%USERPROFILE%\.cache\uv

echo ======================================================================
echo          AI VIDEO CLIPPER ^& LORA CAPTIONER - INSTALLER
echo ======================================================================
echo.

:: Check for uv
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] uv not found. Installing via winget...
    winget install astral-sh.uv --accept-source-agreements --accept-package-agreements
    
    REM SEARCH STRATEGY:
    REM 1. Standard Winget Links (Symlinks)
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
    )
    
    REM 2. Dynamic Winget Package Folder (finding the folder ending in ...uv_... source)
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv*") do (
        if exist "%%D\uv.exe" set "PATH=%%D;%PATH%"
    )
    
    REM 3. Cargo Bin (fallback)
    if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
    
    REM 4. User-specific valid path from previous runs?
    if exist "%LOCALAPPDATA%\uv\uv.exe" (
         set "PATH=%LOCALAPPDATA%\uv;%PATH%"
    )

    REM Verify installation
    uv --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] uv installed but still not found in PATH for this session.
        echo [INFO] Detected location via 'where uv' might require a restart.
        echo [INFO] Please close this window and run install.bat again.
        pause
        exit /b 0
    )
)

:: Argument parsing
set "RESET_VENV=false"
set "NO_PAUSE=false"

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--reset" set "RESET_VENV=true"
if /i "%~1"=="--no-pause" set "NO_PAUSE=true"
shift
goto parse_args
:end_parse_args

echo.
echo [STEP 1/3] Preparing isolated environment (uv)...

if "%RESET_VENV%"=="true" (
    if exist ".venv" (
        echo [INFO] Resetting virtual environment as requested...
        rd /s /q ".venv"
    )
)

if not exist ".venv" (
    uv venv .venv --python 3.10 --seed --managed-python --link-mode hardlink
)

:: --------------------------------------------------------------------
:: [STEP 1.5] FFmpeg Installation & Path Refresh
:: --------------------------------------------------------------------
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] FFmpeg not found. Installing via winget...
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    
    REM ATTEMPT DYNAMIC PATH REFRESH
    REM 1. Standard Winget Links (Symlinks)
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
        echo [INFO] Added Winget Links to PATH for this session.
    )

    REM 2. Robust Search for Gyan.FFmpeg Package (Deep Search)
    REM User Path: %LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-*-full_build\bin
    for /d %%P in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*") do (
        echo [INFO] Found Legacy Package: %%~nxP
        for /d %%B in ("%%P\ffmpeg-*-full_build") do (
             if exist "%%B\bin\ffmpeg.exe" (
                 set "PATH=%%B\bin;%PATH%"
                 echo [INFO] Added Deep FFmpeg Path: %%B\bin
             )
        )
    )
    
    REM Verify
    where ffmpeg >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARNING] FFmpeg installed but not detected in current session.
        echo [IMPORTANT] You may need to RESTART your terminal/PC before running the app.
    ) else (
        echo [SUCCESS] FFmpeg detected!
    )
)

:: Privacy Configuration (On-the-fly)
if not exist ".streamlit\config.toml" (
    echo [INFO] Applying privacy settings ^(Headless Mode + No Analytics^)...
    if not exist ".streamlit" mkdir .streamlit
    (
        echo [browser]
        echo gatherUsageStats = false
        echo.
        echo [server]
        echo headless = true
        echo maxUploadSize = 4096
    ) > .streamlit\config.toml
)

echo .
echo [STEP 2/3] Installing Torch Engine (CUDA 12.8)...
call .venv\Scripts\activate.bat
uv pip install ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --link-mode hardlink ^
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"

echo [INFO] Syncing GGUF High-Performance Backend (CUDA 12.8)...
REM GPU Architecture Detection
set "IS_MODERN_GPU=false"
set "MAJOR_CAP="

for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=compute_cap --format=noheader 2^>nul') do (
    set "MAJOR_CAP=%%a"
)

REM If we got nothing from nvidia-smi, skip GPU-specific logic
if not defined MAJOR_CAP goto gpu_detect_done

echo [INFO] Detected NVIDIA GPU Compute Capability Major: %MAJOR_CAP%

REM Check if MAJOR_CAP is numeric (avoid "N/A" or garbage)
echo(%MAJOR_CAP% | findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 goto gpu_detect_done

REM Now it's safe to compare numerically
if %MAJOR_CAP% GEQ 9 (
    echo [INFO] Modern GPU detected (Hopper/Blackwell). Selecting optimized Blackwell wheel.
    set "IS_MODERN_GPU=true"
)

:gpu_detect_done

if "%IS_MODERN_GPU%"=="false" (
    echo [INFO] Standard GPU detected. Selecting standard universal wheel.
)

if "%IS_MODERN_GPU%"=="true" (
    set "WIN_WHEEL_URL=https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_noavx512_Blackwell-cp310-cp310-win_amd64.whl"
    set "WIN_WHEEL_SHA256=6c13577479d21d51832b2b0f5a75dc64a76ed40ed3f97c9e46bdcf666e286b69"
    set "WHEEL_FILE=llama_cpp_python-0.3.26+cu128_noavx512_Blackwell-cp310-cp310-win_amd64.whl"
) else (
    set "WIN_WHEEL_URL=https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_noavx512-cp310-cp310-win_amd64.whl"
    set "WIN_WHEEL_SHA256=d199417da48fb5158390920aa100a0fac4a5c5139059a3e843dad6a7a6461977"
    set "WHEEL_FILE=llama_cpp_python-0.3.26+cu128_noavx512-cp310-cp310-win_amd64.whl"
)

echo [INFO] Downloading wheel for verification...
curl -L -o "%WHEEL_FILE%" "%WIN_WHEEL_URL%"
if %errorlevel% neq 0 (
    echo [ERROR] Download failed.
    pause
    exit /b 1
)

echo [INFO] Verifying checksum...
certutil -hashfile "%WHEEL_FILE%" SHA256 | findstr /i "%WIN_WHEEL_SHA256%" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Checksum verification failed!
    del "%WHEEL_FILE%"
    pause
    exit /b 1
)

echo [SUCCESS] Checksum verified! Installing...
uv pip install "%WHEEL_FILE%" --force-reinstall
del "%WHEEL_FILE%"


echo.
echo [STEP 3/3] Installing AI Stack...
uv pip install "git+https://github.com/m-bain/whisperX.git@6ec4a020489d904c4f2cd1ed097674232d2692d4" --no-deps --link-mode hardlink

echo [INFO] Ensuring correct CTranslate2 (Windows) - Pinning <4.7.0 to avoid ROCm bug...
uv pip install "ctranslate2<4.7.0" --index-url https://pypi.org/simple --force-reinstall

echo [INFO] Syncing remaining dependencies from pyproject.toml...
uv pip install -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128 --link-mode hardlink

:: --- NOWA SEKCJA v4.0 ---
echo.
echo [STEP 3.5] Installing Audio Intelligence Stack (Qwen2-Audio Support)...
echo [INFO] Adding librosa, soundfile and updating transformers...
uv pip install librosa soundfile numpy --link-mode hardlink
uv pip install --upgrade transformers accelerate huggingface_hub --link-mode hardlink
:: ------------------------

echo.
echo [CHECK] Verifying GPU Acceleration...
call .venv\Scripts\python -c "from llama_cpp import llama_supports_gpu_offload; print(f'>>> GPU Offload Supported: {llama_supports_gpu_offload()}')"

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo You can now run the program using "Run.bat".
echo.
echo.
if "%NO_PAUSE%"=="false" pause