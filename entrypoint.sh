#!/bin/bash
# FNGarvin - AI Video Clipper & LoRA Captioner Entrypoint
# MIT License 2026

# --- SSH Setup ---
SSH_DIR="/root/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [ -n "$PUBLIC_KEY" ]; then
    echo "[INFO] Injecting SSH public key..."
    echo "$PUBLIC_KEY" > "$SSH_DIR/authorized_keys"
    chmod 600 "$SSH_DIR/authorized_keys"
    chown root:root "$SSH_DIR/authorized_keys"
fi

# ==========================================
# GGUF Backend - Dynamic Library Path Fix
# ==========================================
# We must ensure the `llama-cpp-python` backend native libraries can dynamically find the 12.8 CUDA toolkit provided by PyTorch.
# If these paths are missed, the backend will crash on an `str/str` TypeError internal to `load_shared_library()` or core dump.
export NVIDIA_LIBS=$(python3 -c 'import site, os, glob; paths = [glob.glob(os.path.join(p, "nvidia/*/lib")) for p in site.getsitepackages()]; print(":".join([p for sub in paths for p in sub]))' 2>/dev/null)
export LD_LIBRARY_PATH=$NVIDIA_LIBS:$LD_LIBRARY_PATH
echo "[INFO] Injected PyTorch CUDA paths into LD_LIBRARY_PATH."

# Start SSHD
echo "[INFO] Starting SSHD..."
mkdir -p /run/sshd
/usr/sbin/sshd

# --- Runtime GPU Optimization (Hot-Swap) ---
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=noheader,csv | head -n 1)
    echo "[INFO] Detected NVIDIA GPU Compute Capability: $COMPUTE_CAP"
    MAJOR_CAP=$(echo "$COMPUTE_CAP" | cut -d'.' -f1)
    
    if [ "$MAJOR_CAP" -ge 9 ]; then
        echo "[INFO] Modern GPU detected (Hopper/Blackwell). Optimizing AI stack for peak performance..."
        
        BW_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.26+cu128_Blackwell-cp312-cp312-linux_x86_64.whl"
        BW_WHEEL_SHA256="89071f3c7452d24c9442677b7b8bed3d2b1d7ef7a3ca8e05580160aa965cb607"
        TEMP_WHEEL="/tmp/llama_cpp_bw.whl"

        echo "[INFO] Downloading Blackwell optimized wheel..."
        if curl -L -o "$TEMP_WHEEL" "$BW_WHEEL_URL"; then
             echo "$BW_WHEEL_SHA256  $TEMP_WHEEL" | sha256sum -c -
             if [ $? -eq 0 ]; then
                 echo "[INFO] Checksum verified. Hot-swapping llama-cpp-python..."
                 uv pip install --system "$TEMP_WHEEL" --no-deps --force-reinstall
                 echo "[SUCCESS] System optimized for Blackwell/Hopper."
             else
                 echo "[WARNING] Checksum verification failed for optimized wheel. Falling back to standard build."
             fi
             rm -f "$TEMP_WHEEL"
        else
            echo "[WARNING] Failed to download optimized wheel. Using standard build."
        fi
    else
        echo "[INFO] Standard GPU detected. Running with universal build."
    fi
else
    echo "[INFO] No NVIDIA GPU detected via nvidia-smi. Running in CPU/Fallback mode."
fi

# --- Filebrowser Setup ---
echo "[INFO] Starting Filebrowser on port 8080..."
# Start filebrowser in background
nohup /usr/local/bin/filebrowser --address 0.0.0.0 --port 8080 --root /workspace --noauth &> /filebrowser.log &

# --- Main Application ---
echo "[INFO] Starting main application..."
# Execute the original run.sh script with passed arguments
exec ./run.sh "$@"

#EOF entrypoint.sh
