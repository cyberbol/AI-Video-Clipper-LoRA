@echo off
title AI Video Clipper - Updater
echo ---------------------------------------
echo AI Video Clipper ^& LoRA Captioner
echo Checking for updates...
echo ---------------------------------------

:: Sprawdzenie czy git jest zainstalowany
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed. 
    echo Please install Git or download the new version manually from GitHub.
    pause
    exit
)

:: Pobieranie zmian z GitHub
echo Fetching latest changes from GitHub...
git pull origin main

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Project updated! Now refreshing dependencies...
    echo [INFO] This may take a moment. A full log is being written to install.log
    echo.
    call Install.bat --reset --no-pause > install.log 2>&1
    
    if %errorlevel% equ 0 (
        echo [SUCCESS] Update and installation complete!
    ) else (
        echo [ERROR] Dependency refresh failed. Please check install.log for details.
    )
    echo.
) else (
    echo.
    echo [WARNING] Could not update automatically. 
    echo This usually happens if you modified app.py yourself.
    echo.
)

pause
