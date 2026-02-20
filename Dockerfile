# FNGarvin - AI Video Clipper & LoRA Captioner Container
# MIT License 2026

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_HTTP_TIMEOUT=3600
ENV UV_LINK_MODE=hardlink
ENV UV_CACHE_DIR="/root/.cache/uv"
ENV PATH="/root/.local/bin:$PATH"

# Optional Apt Proxy setup
ARG APT_PROXY
RUN if [ -n "$APT_PROXY" ]; then \
    echo "Acquire::http::Proxy \"$APT_PROXY\";" > /etc/apt/apt.conf.d/01proxy; \
    fi

# Install system dependencies (Uses native Podman/Buildah cache mounts for CI/CD)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# Copy only necessary files first to leverage build cache
COPY pyproject.toml .

# Create venv and install dependencies
# We use Python 3.10 as requested in install.sh
# Copy install script
COPY install.sh .
RUN chmod +x install.sh

# Run Installer (Uses cache mount for speed)
# - SKIP_GPU_CHECK: Prevents failure on CPU-only build runners
# - UV_LINK_MODE: 'copy' avoids cross-filesystem hardlink errors/warnings
RUN --mount=type=cache,target=/root/.cache/uv \
    export SKIP_GPU_CHECK=true && \
    export UV_LINK_MODE=copy && \
    ./install.sh --system


# --- MULTI-SERVICE INTEGRATION (Appended to preserve cache) ---

# Install SSHD and Filebrowser
RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

# Configure SSHD (Key-based only)
RUN mkdir -p /run/sshd && \
    sed -i 's/^#\?PermitRootLogin .*$/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication .*$/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo 'ClientAliveInterval 30' >> /etc/ssh/sshd_config && \
    echo 'ClientAliveCountMax 5' >> /etc/ssh/sshd_config

# Copy entrypoint script
COPY entrypoint.sh /workspace/entrypoint.sh

# --- FINAL APP SETUP ---

# Copy the rest of the application
COPY . .

# Apply privacy settings and Runpod Proxy fixes (CORS/XSRF)
RUN mkdir -p .streamlit && \
    echo "[browser]\ngatherUsageStats = false\n[server]\nheadless = true\nmaxUploadSize = 4096\nenableCORS = false\nenableXsrfProtection = false" > .streamlit/config.toml

# Set up local model paths
ENV HF_HOME="/workspace/models"
ENV TORCH_HOME="/workspace/models"
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Make scripts executable
RUN chmod +x run.sh install.sh entrypoint.sh

# Expose Streamlit (8501), Filebrowser (8080), and SSH (22)
EXPOSE 8501 8080 22

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["-h", "0.0.0.0"]

#EOF Dockerfile
