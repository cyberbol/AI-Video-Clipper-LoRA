
@echo off
setlocal EnableDelayedExpansion
:: AI Video Clipper ^& LoRA Captioner - Windows Installer

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

echo.
echo [STEP 1/3] Preparing isolated environment (uv)...
    uv venv .venv --python 3.10 --seed --managed-python --link-mode hardlink

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

echo.
echo [STEP 2/3] Installing Torch Engine (CUDA 12.8)...
call .venv\Scripts\activate.bat
uv pip install ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --link-mode hardlink ^
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"

echo [INFO] Syncing GGUF High-Performance Backend (CUDA 12.8)...
set "FOUND_WHEEL="

REM Robust search using DIR to find the wheel
for /f "delims=" %%f in ('dir /b /s wheels\llama_cpp_python*win_amd64.whl 2^>nul') do (
    set "FOUND_WHEEL=%%f"
)

if not defined FOUND_WHEEL (
    echo.
    echo [ERROR] Pre-built Windows Wheel for llama-cpp-python NOT FOUND!
    echo [ERROR] We require a specific CUDA 12.8 wheel for Python 3.10.
    echo.
    pause
    exit /b 1
)

echo [INFO] Installing verified local CUDA 12.8 wheel: !FOUND_WHEEL!
uv pip install "!FOUND_WHEEL!" --force-reinstall


echo.
echo [STEP 3/3] Installing AI Stack...
uv pip install "git+https://github.com/m-bain/whisperX.git" --no-deps --link-mode hardlink

echo [INFO] Syncing remaining dependencies from pyproject.toml...
uv pip install -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128 --link-mode hardlink

echo.
echo [CHECK] Verifying GPU Acceleration...
.venv\Scripts\python -c "from llama_cpp import llama_supports_gpu_offload; print(f'>>> GPU Offload Supported: {llama_supports_gpu_offload()}')"

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo You can now run the program using "Run.bat".
echo.
pause

::EOF install.bat