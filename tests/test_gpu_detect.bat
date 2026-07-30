@echo off
setlocal EnableDelayedExpansion

:: --------------------------------------------------------------------
:: Regression test for Install.bat's "--detect-only" hook (see issue #13:
:: a GPU-branch code path once crashed cmd.exe's parser outright with
:: ". was unexpected at this time.").
::
:: Since the wheel rebuild in docs/BUILD_WHEELS_HOWTO.md, wheel selection
:: no longer branches on GPU generation - a single universal wheel now
:: covers every supported CUDA architecture. This test just guards
:: against Install.bat crashing regardless of what nvidia-smi reports
:: (missing, garbage output, or a real GPU) and confirms WHEEL_FILE
:: resolves to the expected constant.
::
:: This drives the REAL Install.bat via its "--detect-only" hook (which
:: resolves WHEEL_FILE/WIN_WHEEL_URL and exits immediately, before
:: touching the venv, network, or doing any installs) against a fake
:: nvidia-smi.bat (tests\fixtures\nvidia-smi.bat) so it can be run on
:: any machine, with or without an NVIDIA GPU, in a few seconds.
::
:: Usage: tests\test_gpu_detect.bat
:: Requires: uv on PATH (Install.bat checks for it before anything else).
:: --------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "PATH=%SCRIPT_DIR%fixtures;%PATH%"

set "PASS_COUNT=0"
set "FAIL_COUNT=0"

set "EXPECT_WHEEL=llama_cpp_python-0.3.44+cu128-cp310-cp310-win_amd64.whl"

echo ======================================================================
echo   WHEEL RESOLUTION REGRESSION TEST (Install.bat --detect-only)
echo ======================================================================

call :run_case "RTX 5090 (Blackwell, cap 12.0)"     "12.0"
call :run_case "RTX 4090 (Ada, cap 8.9)"             "8.9"
call :run_case "nvidia-smi returns garbage (N/A)"    "N/A"
call :run_case "No NVIDIA GPU / nvidia-smi missing"  "NONE"

echo.
echo ======================================================================
echo   RESULTS: !PASS_COUNT! passed, !FAIL_COUNT! failed
echo ======================================================================

if !FAIL_COUNT! GTR 0 exit /b 1
exit /b 0

:: ---------------------------------------------------------------
:: %1 = human label, %2 = FAKE_COMPUTE_CAP value
:: ---------------------------------------------------------------
:run_case
set "CASE_LABEL=%~1"
set "FAKE_COMPUTE_CAP=%~2"

set "OUT_FILE=%TEMP%\gpu_detect_test_%RANDOM%.log"
call "%REPO_ROOT%\Install.bat" --detect-only --no-pause > "%OUT_FILE%" 2>&1

set "GOT_WHEEL="
for /f "tokens=2 delims==" %%v in ('findstr /c:"] WHEEL_FILE=" "%OUT_FILE%"') do set "GOT_WHEEL=%%v"

set "PARSE_ERROR="
findstr /c:"was unexpected at this time" "%OUT_FILE%" >nul
if not errorlevel 1 set "PARSE_ERROR=1"
findstr /c:"is not recognized as an internal or external command" "%OUT_FILE%" >nul
if not errorlevel 1 set "PARSE_ERROR=1"
findstr /c:"The syntax of the command is incorrect" "%OUT_FILE%" >nul
if not errorlevel 1 set "PARSE_ERROR=1"

set "CASE_FAILED="
if defined PARSE_ERROR set "CASE_FAILED=1"
if not "!GOT_WHEEL!"=="!EXPECT_WHEEL!" set "CASE_FAILED=1"

if defined CASE_FAILED goto case_fail

set /a PASS_COUNT+=1
echo [PASS] !CASE_LABEL!
del "%OUT_FILE%" >nul 2>&1
goto :eof

:case_fail
set /a FAIL_COUNT+=1
echo [FAIL] !CASE_LABEL!
if defined PARSE_ERROR echo        cmd.exe parser error detected in output - see !OUT_FILE!
if not "!GOT_WHEEL!"=="!EXPECT_WHEEL!" echo        WHEEL_FILE: expected "!EXPECT_WHEEL!", got "!GOT_WHEEL!"
echo        Full output kept at: !OUT_FILE!
goto :eof
