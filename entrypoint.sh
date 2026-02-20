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

# Start SSHD
echo "[INFO] Starting SSHD..."
mkdir -p /run/sshd
/usr/sbin/sshd

# --- Filebrowser Setup ---
echo "[INFO] Starting Filebrowser on port 8080..."
# Start filebrowser in background
nohup /usr/local/bin/filebrowser --address 0.0.0.0 --port 8080 --root /workspace --noauth &> /filebrowser.log &

# --- Main Application ---
echo "[INFO] Starting main application..."
# Execute the original run.sh script with passed arguments
exec ./run.sh "$@"

#EOF entrypoint.sh
