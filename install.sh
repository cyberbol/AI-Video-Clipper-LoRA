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
INSTALL_ARGS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reset) RESET_VENV=true ;;
        --system) 
            USE_SYSTEM=true 
            INSTALL_ARGS="--system --break-system-packages"
            ;;
    esac
    shift
done

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
        uv venv .venv --python 3.10 --seed --managed-python --link-mode hardlink
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

# Detect Python Version
if [ "$USE_SYSTEM" = true ]; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
else
    PY_VER=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
fi

echo "[INFO] Detected Python Version: $PY_VER"

# GPU Architecture Detection
IS_MODERN_GPU=false
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=noheader,csv | head -n 1)
    echo "[INFO] Detected NVIDIA GPU Compute Capability: $COMPUTE_CAP"
    # Parse Major version
    MAJOR_CAP=$(echo "$COMPUTE_CAP" | cut -d'.' -f1)
    if [ "$MAJOR_CAP" -ge 9 ]; then
        echo "[INFO] Modern GPU detected (Hopper/Blackwell). Selecting optimized Sm90+ wheel."
        IS_MODERN_GPU=true
    else
        echo "[INFO] Legacy GPU detected. Selecting standard AVX2/No-AVX512 wheel."
    fi
else
    echo "[WARNING] nvidia-smi not found. Falling back to standard universal wheel."
fi

if [ "$PY_VER" == "3.10" ]; then
    if [ "$IS_MODERN_GPU" = true ]; then
        LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_noavx512_Blackwell-cp310-cp310-linux_x86_64.whl"
        LINUX_WHEEL_SHA256="58d92a5bba695b77699d5c72857fefd67629ae46f37949b9836ca240663409fc"
        WHEEL_FILE="llama_cpp_python-0.3.26+cu128_noavx512_Blackwell-cp310-cp310-linux_x86_64.whl"
    else
        LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_noavx512-cp310-cp310-linux_x86_64.whl"
        LINUX_WHEEL_SHA256="f2fb93f1f859e7ced807b2af2812056510e748de9633ec166b9371230e79836f"
        WHEEL_FILE="llama_cpp_python-0.3.26+cu128_noavx512-cp310-cp310-linux_x86_64.whl"
    fi
elif [ "$PY_VER" == "3.12" ]; then
    if [ "$IS_MODERN_GPU" = true ]; then
        LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_noavx512_Blackwell-cp312-cp312-linux_x86_64.whl"
        LINUX_WHEEL_SHA256="fb26819470e13666112064f016411a74b4e192841bebe8ac7040e6b38d9ffac5"
        WHEEL_FILE="llama_cpp_python-0.3.26+cu128_noavx512_Blackwell-cp312-cp312-linux_x86_64.whl"
    else
        LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_noavx512-cp312-cp312-linux_x86_64.whl"
        LINUX_WHEEL_SHA256="5fbfd0c8073286219cd56ce7c21538134c1fd1cebc5b215576d457bcb941dc1c"
        WHEEL_FILE="llama_cpp_python-0.3.26+cu128_noavx512-cp312-cp312-linux_x86_64.whl"
    fi
else
    echo "[ERROR] Unsupported Python Version for GPU Acceleration: $PY_VER. Only 3.10 and 3.12 supported."
    # Fail hard to prevent broken installs
    exit 1
fi

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
uv pip install $INSTALL_ARGS --upgrade transformers accelerate huggingface_hub --link-mode hardlink


echo ""
if [ "$SKIP_GPU_CHECK" != "true" ]; then
    echo "[CHECK] Verifying GPU Acceleration (Llama CPP)..."
    
    # Set LD_LIBRARY_PATH to include NVIDIA libraries for wheel compatibility checks
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

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

    if [ "$USE_SYSTEM" = true ]; then
        python3 -c "from llama_cpp import llama_supports_gpu_offload; print(f'>>> GPU Offload Supported: {llama_supports_gpu_offload()}')" || echo "WARNING: Llama check failed"
    else
        .venv/bin/python -c "from llama_cpp import llama_supports_gpu_offload; print(f'>>> GPU Offload Supported: {llama_supports_gpu_offload()}')" || echo "WARNING: Llama check failed"
    fi
else
    echo "[INFO] Skipping GPU Verification (Build Mode)"
fi

echo "======================================================================"
echo "Installation complete!"
echo "Run the app with: ./run.sh"

