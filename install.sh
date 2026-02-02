#!/bin/bash
# AI Video Clipper & LoRA Captioner - Linux/WSL Installer

# Exit on error
set -e

# UV Optimizations
export UV_HTTP_TIMEOUT=3600
export UV_LINK_MODE=hardlink
export UV_CACHE_DIR="${HOME}/.cache/uv"

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

echo "[STEP 1/3] Preparing Environment..."
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.10 --link-mode hardlink
fi
source .venv/bin/activate

# Privacy Configuration (On-the-fly)
if [ ! -f ".streamlit/config.toml" ]; then
    echo "[INFO] Applying privacy settings (Headless Mode + No Analytics)..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml <<EOL
[browser]
gatherUsageStats = false
[server]
headless = true
EOL
fi

echo "[STEP 2/3] Installing Torch Engine (CUDA 12.8)..."
uv pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --link-mode hardlink \
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"

echo "[STEP 3/3] Installing AI Stack..."
uv pip install \
    --link-mode hardlink \
    "git+https://github.com/m-bain/whisperX.git" --no-deps

echo "[INFO] Syncing remaining dependencies from pyproject.toml..."
uv pip install \
    --link-mode hardlink \
    -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128

echo "======================================================================"
echo "Installation complete!"
echo "Run the app with: ./run.sh"

