#!/bin/bash
# --------------------------------------------------------------------
# Regression test for install.sh's --detect-only hook (Linux/WSL
# install-time path). Mirrors tests/test_gpu_detect.bat for Windows
# Install.bat; see issue #13 for the cmd.exe-specific bug this whole
# effort started from.
#
# Since the wheel rebuild in docs/BUILD_WHEELS_HOWTO.md, wheel selection
# no longer branches on GPU generation - one universal wheel per Python
# version now covers every supported CUDA architecture. This test just
# guards against install.sh crashing regardless of what nvidia-smi
# reports (missing, garbage output, or a real GPU) and confirms
# WHEEL_FILE tracks the detected Python version correctly.
#
# Drives the REAL install.sh via its --detect-only hook against a fake
# nvidia-smi (tests/fixtures/nvidia-smi) so it runs in under a second,
# with or without an NVIDIA GPU, and without ffmpeg/uv installed.
#
# Usage: tests/test_gpu_detect.sh
# --------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
export PATH="$SCRIPT_DIR/fixtures:$PATH"

PASS_COUNT=0
FAIL_COUNT=0

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
if [ "$PY_VER" == "3.10" ]; then
    EXPECT_WHEEL="llama_cpp_python-0.3.44+cu128-cp310-cp310-linux_x86_64.whl"
else
    EXPECT_WHEEL="llama_cpp_python-0.3.44+cu128-cp312-cp312-linux_x86_64.whl"
fi

echo "======================================================================"
echo "  WHEEL RESOLUTION REGRESSION TEST (install.sh --detect-only)"
echo "  local python3 resolves to $PY_VER"
echo "======================================================================"

run_case() {
    local label="$1" cap="$2"
    local out_file
    out_file="$(mktemp)"

    FAKE_COMPUTE_CAP="$cap" bash "$REPO_ROOT/install.sh" --detect-only > "$out_file" 2>&1

    local got_wheel
    got_wheel=$(grep -m1 "\] WHEEL_FILE=" "$out_file" | cut -d= -f2)

    local failed=""
    if grep -qi "unbound variable\|command not found\|integer expression expected\|Traceback" "$out_file"; then
        failed="1"
    fi
    if [ "$got_wheel" != "$EXPECT_WHEEL" ]; then failed="1"; fi

    if [ -n "$failed" ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "[FAIL] $label"
        echo "       WHEEL_FILE: expected '$EXPECT_WHEEL', got '$got_wheel'"
        echo "       Full output kept at: $out_file"
    else
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "[PASS] $label"
        rm -f "$out_file"
    fi
}

run_case "RTX 5090 (Blackwell, cap 12.0)"     "12.0"
run_case "RTX 4090 (Ada, cap 8.9)"            "8.9"
run_case "nvidia-smi returns garbage (N/A)"   "N/A"
run_case "No NVIDIA GPU / nvidia-smi missing" "NONE"

echo
echo "======================================================================"
echo "  RESULTS: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "======================================================================"

[ "$FAIL_COUNT" -eq 0 ]
