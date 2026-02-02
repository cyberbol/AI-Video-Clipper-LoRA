@echo off
:: AI Video Clipper ^& LoRA Captioner Launcher

TITLE AI Clipper Launcher
color 0A

:: Defaults
set PORT=8501
set HOST=127.0.0.1

:: Parse Args (Simple Loop)
:parse
if "%~1"=="" goto endparse
if "%~1"=="-p" (
    set PORT=%~2
    shift
    shift
    goto parse
)
if "%~1"=="-h" (
    set HOST=%~2
    shift
    shift
    goto parse
)
shift
goto parse
:endparse

:: Set up local model paths
set HF_HOME=.\models
set TORCH_HOME=.\models
set KMP_DUPLICATE_LIB_OK=TRUE

:: Activate environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Starting Streamlit on %HOST%:%PORT%...

streamlit run app.py --server.port %PORT% --server.address %HOST%

pause

::EOF Run.bat
