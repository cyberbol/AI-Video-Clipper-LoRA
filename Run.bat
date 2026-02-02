@echo off
TITLE AI Video Clipper - Created by Cyberbol
color 0B

echo ======================================================
echo      STARTING AI VIDEO CLIPPER (Created by Cyberbol)
echo ======================================================
echo.

if not exist venv (
    echo [ERROR] Environment not found! Please run install.bat first.
    pause
    exit
)

:: --- ENVIRONMENT CONFIG (LOCAL MODELS) ---
:: Tworzymy folder models jesli nie istnieje
if not exist models mkdir models
:: Ustawiamy zmienną HF_HOME na folder models w katalogu programu
set HF_HOME=%~dp0models
:: Opcjonalnie mozna tez przestawic cache Torcha
set TORCH_HOME=%~dp0models\torch

call venv\Scripts\activate.bat

echo Launching Interface...
streamlit run app.py

pause