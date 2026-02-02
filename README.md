# 👁️🐧 AI Video Clipper & LoRA Captioner

![Demo](demo.webp)

## 📖 Description
This tool is a powerful automation suite designed for creating **AI training datasets (LoRA/Fine-tuning)**. 
It automatically detects speech segments in long videos using **WhisperX**, clips them, and generates detailed captions combining:
1.  **Audio Transcription** (what the person says).
2.  **Visual Description** (what the scene looks like) using **Qwen2-VL**.

It includes **two Vision Models** that you can switch between inside the app:
* **Qwen2-VL-7B:** High quality, detailed descriptions (Requires ~16GB VRAM/RAM loading).
* **Qwen2-VL-2B:** Extremely fast, slightly less detailed (Requires ~5GB VRAM/RAM loading).

---

## ⚙️ Prerequisites
- **Git**
- **FFmpeg** (Included in Windows installer; requires `sudo apt install ffmpeg` on Linux)
- **NVIDIA Drivers** (CUDA 12.8+)
- **Internet Connection** (for initial model and environment setup)

*Note: This project uses `uv` to automatically manage its own Python 3.10 environment and dependencies. You do not need to install Python or Build Tools manually.*

---

## 🚀 Installation

### Windows
1. Double-click **`install.bat`**.
2. Once finished, use **`Run.bat`** to start the app.

### Linux / WSL
1. Open a terminal in the project folder.
2. Run: `chmod +x install.sh run.sh`
3. Run: `./install.sh`
4. Once finished, start the app with: `./run.sh`

---

## ▶️ How to Run
1. **Launch the App:**
   **Linux / WSL:**
   ```bash
   ./run.sh
   # Optional: ./run.sh -p 9000 -h 0.0.0.0
   ```
   **Windows:**
   Double-click `Run.bat` or run via terminal:
   ```cmd
   Run.bat
   :: Optional: Run.bat -p 9000 -h 0.0.0.0
   ```
2. A web interface will open in your browser.
3. **Select Your Model:** Select **7B (Quality)** or **2B (Speed)** in the sidebar.
4. **Upload a Video** (MP4/MKV).
5. **Configure Settings:**
    * **Target Duration:** Desired length of clips (e.g., 5-10 seconds).
    * **Trigger Word:** (Optional) Enter your training token here (e.g., `prstxxx`).
6. Click **`START PROCESS 🚀`**.

> [!NOTE]
> On the first run, the necessary models (Whisper, Qwen) will be downloaded. This may take some time depending on your connection.

---

## 📂 Output
* Generated clips and captions are saved in the **`dataset`** folder.
* Files are named sequentially (e.g., `001.mp4`, `001.txt`).

---

<div align="center">
  <b>Project maintained by Cyberbol</b>
</div>
