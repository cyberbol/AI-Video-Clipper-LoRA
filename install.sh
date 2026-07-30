#!/bin/bash
# AI Video Clipper & LoRA Captioner - Linux/WSL Installer

# Exit on error
set -e

# UV Optimizations
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-3600}
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"

echo "======================================================================"
echo "         AI VIDEO CLIPPER & LORA CAPTIONER - INSTALLER (Linux/WSL)"
echo "======================================================================"

# Check for FFmpeg (Linux)
if ! command -v ffmpeg &> /dev/null; then
    echo "[ERROR] FFmpeg is missing!"
    echo "This tool requires FFmpeg to process video/audio."
    echo "Please install it using your package manager, e.g.:"
    echo "  sudo apt update && sudo apt install -y ffmpeg"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin"
fi

# Argument parsing
RESET_VENV=false
USE_SYSTEM=false
DETECT_ONLY=false
INSTALL_ARGS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reset) RESET_VENV=true ;;
        --system)
            USE_SYSTEM=true
            INSTALL_ARGS="--system --break-system-packages"
            ;;
        --detect-only) DETECT_ONLY=true ;;
    esac
    shift
done

# --------------------------------------------------------------------
# llama-cpp-python wheel resolution.
#
# As of the wheel rebuild in docs/BUILD_WHEELS_HOWTO.md, a single wheel
# per Python version covers every supported CUDA architecture (Turing
# through Blackwell) - there is no more per-GPU-family branching here.
# The compute-cap probe below is diagnostic-only (printed for
# troubleshooting) and does not affect which wheel gets fetched.
#
# Sets: PY_VER, WHEEL_FILE, LINUX_WHEEL_URL, LINUX_WHEEL_SHA256
# --------------------------------------------------------------------
resolve_wheel() {
    if [ "$USE_SYSTEM" = true ] || [ ! -x ".venv/bin/python" ]; then
        PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    else
        PY_VER=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    fi
    echo "[INFO] Detected Python Version: $PY_VER"

    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
        echo "[INFO] Detected NVIDIA GPU Compute Capability: $COMPUTE_CAP"
    fi

    if [ "$PY_VER" == "3.10" ]; then
        LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.4-llama-deps/llama_cpp_python-0.3.44+cu128-cp310-cp310-linux_x86_64.whl"
        LINUX_WHEEL_SHA256="6cf11a799a54b29aebc2f3d4a436d61844516a475045c130fee74814852f075f"
        WHEEL_FILE="llama_cpp_python-0.3.44+cu128-cp310-cp310-linux_x86_64.whl"
    elif [ "$PY_VER" == "3.12" ]; then
        LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.4-llama-deps/llama_cpp_python-0.3.44+cu128-cp312-cp312-linux_x86_64.whl"
        LINUX_WHEEL_SHA256="baec17ea25494ab79c998befba14a2f5960c17838dc27453d7c10b1da222f6fa"
        WHEEL_FILE="llama_cpp_python-0.3.44+cu128-cp312-cp312-linux_x86_64.whl"
    else
        echo "[ERROR] Unsupported Python Version for GPU Acceleration: $PY_VER. Only 3.10 and 3.12 supported."
        # Fail hard to prevent broken installs
        exit 1
    fi
}

# Test hook: resolve the wheel to fetch and exit immediately, before
# touching ffmpeg/uv checks, the venv, network, or any installs. Used by
# tests/test_gpu_detect.sh to verify this doesn't crash regardless of
# GPU presence.
if [ "$DETECT_ONLY" = true ]; then
    resolve_wheel
    echo "[DETECT-ONLY] PY_VER=$PY_VER"
    echo "[DETECT-ONLY] WHEEL_FILE=$WHEEL_FILE"
    echo "[DETECT-ONLY] LINUX_WHEEL_URL=$LINUX_WHEEL_URL"
    echo "[DETECT-ONLY] LINUX_WHEEL_SHA256=$LINUX_WHEEL_SHA256"
    exit 0
fi

echo "[STEP 1/3] Preparing Environment..."

if [ "$USE_SYSTEM" = true ]; then
    echo "[INFO] Using system Python environment (Skipping venv creation)..."
    # We assume python3 is available in the base image
else
    if [ "$RESET_VENV" = true ]; then
        if [ -d ".venv" ]; then
            echo "[INFO] Resetting virtual environment as requested..."
            rm -rf .venv
        fi
    fi

    if [ ! -d ".venv" ]; then
        # --managed-python doesn't exist on uv 0.5.21 (what this bootstrap
        # leaves in place when uv is already on PATH, e.g. inside the
        # container image). --python-preference only-managed is the 0.5.21
        # equivalent: forces uv's own downloaded interpreter so the venv
        # matches our pinned llama-cpp-python wheel ABI and never touches
        # whatever Python happens to already be on the user's system.
        uv venv .venv --python 3.10 --seed --python-preference only-managed --link-mode hardlink
    fi
    source .venv/bin/activate
fi

# Privacy Configuration (On-the-fly)
if [ ! -f ".streamlit/config.toml" ]; then
    echo "[INFO] Applying privacy settings (Headless Mode + No Analytics)..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml <<EOL
[browser]
gatherUsageStats = false
[server]
headless = true
maxUploadSize = 4096
EOL
fi

echo "[STEP 2/3] Installing Torch Engine (CUDA 12.8)..."
# Skip Torch install if using system and it's likely present (Docker base image)
if [ "$USE_SYSTEM" = true ] && python3 -c "import torch" &> /dev/null; then
    echo "[INFO] System Torch detected. Skipping explicit Torch installation."
else
    uv pip install $INSTALL_ARGS \
        --index-url https://download.pytorch.org/whl/cu128 \
        --link-mode hardlink \
        "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"
fi

echo "[STEP 3/3] Installing AI Stack..."
uv pip install $INSTALL_ARGS \
    --link-mode hardlink \
    "git+https://github.com/m-bain/whisperX.git@6ec4a020489d904c4f2cd1ed097674232d2692d4" --no-deps

echo "[INFO] Syncing GGUF High-Performance Backend (CUDA 12.8)..."
resolve_wheel

echo "[INFO] Downloading wheel for verification..."
curl -L -o "$WHEEL_FILE" "$LINUX_WHEEL_URL"

echo "[INFO] Verifying checksum..."
echo "$LINUX_WHEEL_SHA256  $WHEEL_FILE" | sha256sum -c -

if [ $? -ne 0 ]; then
    echo "[ERROR] Checksum verification failed!"
    rm "$WHEEL_FILE"
    exit 1
fi

echo "[INFO] Checksum verified! Installing..."
uv pip install $INSTALL_ARGS "$WHEEL_FILE" --force-reinstall
rm "$WHEEL_FILE"


# Fix for ROCm/Linux compatibility or just general stability matching Windows
echo "[INFO] Ensuring correct CTranslate2 - Pinning <4.7.0..."
uv pip install $INSTALL_ARGS "ctranslate2<4.7.0" --index-url https://pypi.org/simple --force-reinstall

echo "[INFO] Syncing basic dependencies from pyproject.toml..."
uv pip install $INSTALL_ARGS \
    --link-mode hardlink \
    -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[STEP 3.5] Installing Audio Intelligence Stack (Qwen2-Audio Support)..."
echo "[INFO] Adding librosa, soundfile and updating transformers..."
uv pip install $INSTALL_ARGS librosa soundfile --link-mode hardlink
uv pip install $INSTALL_ARGS transformers accelerate huggingface_hub --link-mode hardlink


echo ""
if [ "$SKIP_GPU_CHECK" != "true" ]; then
    echo "[CHECK] Verifying GPU Acceleration (Llama CPP)..."
    
    # We rely on the dynamic detector below for the exact nvidia lib path

# Logic to pick the correct wheel based on OS and Python version
    if [ "$USE_SYSTEM" = true ]; then
        SITE_PACKAGES=$(python3 -m site --user-site 2>/dev/null)
        [ -z "$SITE_PACKAGES" ] && SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    else
        SITE_PACKAGES=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    fi

    if [ -d "$SITE_PACKAGES/nvidia" ]; then
        # Find all directories named 'lib' under nvidia/ and join them into a path string
        LIB_PATHS=$(find "$SITE_PACKAGES/nvidia" -type d -name "lib" 2>/dev/null | paste -sd ":" - || echo "")
        if [ -n "$LIB_PATHS" ]; then
            export LD_LIBRARY_PATH="$LIB_PATHS:$LD_LIBRARY_PATH"
        fi
    fi

    # llama_supports_gpu_offload() checks a static build-time flag that no
    # longer means anything on GGML_BACKEND_DL wheels (CUDA ships as a
    # separate dynamically-loaded backend, not compiled into libllama.so) -
    # it reports False unconditionally regardless of whether CUDA actually
    # loads. Load the backends the same way Llama.__init__ does and check
    # for a real GPU device instead.
    GPU_CHECK_PY="
import ctypes
from pathlib import Path
import llama_cpp.llama_cpp as llama_cpp_lib
from llama_cpp._ggml import ggml_backend_load_all_from_path, ggml_backend_dev_by_type
lib_dir = Path(llama_cpp_lib.__file__).resolve().parent / 'lib'
ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode('utf-8')))
print(f'>>> GPU Offload Supported: {bool(ggml_backend_dev_by_type(1))}')
"
    # llama_cpp prints a wall of native "loaded library from ..." /
    # "optional API unavailable" / ggml backend-init noise on every fresh
    # process (see modules/vision_engine.py for the equivalent runtime fix) -
    # this is a one-shot subprocess so that fix doesn't reach it. Capture
    # everything and surface only the one line we actually care about;
    # dump the full output if that line is missing (real failure).
    if [ "$USE_SYSTEM" = true ]; then
        GPU_CHECK_OUTPUT=$(python3 -c "$GPU_CHECK_PY" 2>&1)
    else
        GPU_CHECK_OUTPUT=$(.venv/bin/python -c "$GPU_CHECK_PY" 2>&1)
    fi
    GPU_CHECK_RESULT=$(echo "$GPU_CHECK_OUTPUT" | grep ">>> GPU Offload Supported:")
    if [ -n "$GPU_CHECK_RESULT" ]; then
        echo "$GPU_CHECK_RESULT"
    else
        echo "WARNING: Llama GPU check failed to run"
        echo "$GPU_CHECK_OUTPUT"
    fi
else
    echo "[INFO] Skipping GPU Verification (Build Mode)"
fi

echo "======================================================================"
echo "Installation complete!"
echo "Run the app with: ./run.sh"

