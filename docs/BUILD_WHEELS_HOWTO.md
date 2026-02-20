# Building CUDA-Accelerated llama-cpp-python Wheels

This guide provides the workflow for manually building `llama-cpp-python` wheels with CUDA support. This is required when upgrading the CUDA runtime (e.g., from 12.8) or supporting newer model architectures not covered by standard releases.

## Prerequisites

* **NVIDIA CUDA Toolkit:** Installed (the version MUST match the one torch et al. are using) and `nvcc` available in the system PATH.  You can download it from [here](https://developer.nvidia.com/cuda-downloads).
* **C++ Compiler:** * **Windows:** [Visual Studio 2022 Community with C++ workloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    * **Linux:** `gcc`, `g++`, and `cmake`.
* **Python:** 3.10+ (or whatever the project is using at the time).

---

## Windows Build Process

Perform these steps in a **Developer PowerShell for VS 2022** console.

### 1. Environment Preparation
```powershell
# Clone the project and enter the directory
git clone https://github.com/cyberbol/AI-Video-Clipper-LoRA.git
cd AI-Video-Clipper-LoRA

# Run the project installer to establish the base environment
.\install.bat
# Don't worry if it doesn't finish to completion.  The key is that it installs the virtual environment and ensures your development environment is prepared for the correct version of Python, etc.


# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Compile the Wheel
```powershell
# Ensure the cache is clear for this package
uv cache clean llama-cpp-python

# Install Build Requirements (Ninja avoids the slow, sequential MSBuild generator!)
uv pip install ninja scikit-build-core cmake

# Set Build Environment Variables
# Tell CMake to use Ninja instead of the slow MSBuild
$env:CMAKE_GENERATOR = "Ninja"
$env:FORCE_CMAKE = "1"

# IMPORTANT:Change this to your CUDA installation path
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" 

# Instruct the MSVC C/C++ compiler to use ALL available CPU cores
$env:CFLAGS = "/MP"
$env:CXXFLAGS = "/MP"

# The number of parallel build jobs (Set to your CPU Thread Count)
$env:CMAKE_BUILD_PARALLEL_LEVEL = "16"

# Compiler Args - Disable AVX512 for broad compatibility unless you have a modern workstation CPU
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_BUILD_TYPE=Release -DLLAMA_AVX512=OFF"

# Build the wheel from the JamePeng fork (or latest supported source)
pip wheel git+https://github.com/JamePeng/llama-cpp-python.git@main --no-deps --wheel-dir=wheels --no-cache-dir

# Rename the artifact to indicate CUDA support
# Note: Ensure the filename matches the version generated in the \wheels folder
ren wheels\llama_cpp_python-0.3.16-cp310-cp310-win_amd64.whl llama_cpp_python-0.3.16+cu128-cp310-cp310-win_amd64.whl
```

---

## Linux Build Process

### 1. Environment Preparation
```bash
#use whatever version of python matches the current base image
uv venv .venv --python 3.12 --seed --managed-python --link-mode hardlink
source .venv/bin/activate
```

### 2. Compile the Wheel
```bash
# Ensure the cache is clear
uv cache clean llama-cpp-python

# Build using inline environment variables
# IMPORTANT: Use -DLLAMA_AVX512=OFF to prevent "Illegal instruction" on non-AVX512 CPUs (like many Runpod nodes)
# IMPORTANT: If targeting RTX 5090 or other Blackwell GPUs, you MUST include -DCMAKE_CUDA_ARCHITECTURES=90
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export FORCE_CMAKE=1
export CUDA_PATH=/usr/local/cuda
export CMAKE_ARGS="-DGGML_CUDA=ON -DLLAMA_AVX512=OFF -DCMAKE_CUDA_ARCHITECTURES=all -DCMAKE_BUILD_TYPE=Release"
export CMAKE_BUILD_PARALLEL_LEVEL=8 
pip wheel git+https://github.com/JamePeng/llama-cpp-python.git --no-deps --wheel-dir=wheels --no-cache-dir

# Rename the artifact to indicate the Universal/AVX2 status
# The resulting wheel will be compatible with almost all x86_64 CPUs from the last 10 years.
mv wheels/llama_cpp_python-0.3.26-cp312-cp312-linux_x86_64.whl wheels/llama_cpp_python-0.3.26+cu128-avx2-cp312-cp312-linux_x86_64.whl
```

---

## Deployment
1. Upload the renamed `.whl` files to a GitHub Release.
2. Update `install.sh` and `install.bat` to reference the new download URLs for these specific wheels.

// END OF BUILDING_WHEELS.md