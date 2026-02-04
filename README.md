# 👁️🐧👂 AI Video Clipper & LoRA Captioner (v4.0)

**The ultimate local dataset preparation tool for Video LoRA training (LTX-2, HunyuanVideo).**
*Now featuring Audio Intelligence, 3-Stage Pipeline, and Advanced Bulk Processing.*

---

## ⚡ What's New in v4.0 (Audio Intelligence Update)

We have completely rewritten the core engine to support multi-modal understanding.

* **🎧 Audio Intelligence (Qwen2-Audio Support):**
    The app can now "listen" to your videos! It uses the **Qwen2-Audio-7B-Instruct** model to analyze background sounds, ambiance, and music (e.g., *"In the background, wind is blowing and melancholic music is playing"*). This is crucial for training advanced video models like LTX-Video that support audio generation.
* **🚀 3-Stage Pipeline Architecture:**
    To manage VRAM usage efficiently, the app now operates in strict phases:
    1.  **Speech Analysis:** (WhisperX) - Precise timestamping.
    2.  **Audio Event Analysis:** (Qwen2-Audio) - Description of environmental sounds.
    3.  **Visual Analysis:** (Qwen2-VL) - Detailed visual captioning.
    *Memory is aggressively cleaned between stages to prevent OOM errors.*
* **📦 Bulk Video Captioner 2.0:**
    Completely revamped! You can now toggle **Vision Captioning** and **Speech Transcription** independently. Process folders of existing clips with full AI power.
* **⚡ LTX Hard Cut Mode:**
    A new cutting mode designed for LTX-2 training. It ignores sentence boundaries and cuts clips to a strict duration (e.g., exactly 5.0s) from the start of detected speech, ensuring perfect tensor shapes for training.
* **🌊 Natural Flow Captions:**
    Rewritten prompt logic merges Visual descriptions, Audio atmosphere, and Speech into one fluid, natural paragraph instead of robotic tags.

---

## 🎯 Core Features

### 1. 🎥 Video Auto-Clipper
Upload a long video (e.g., a vlog or podcast). The AI will:
* Detect speech using **WhisperX** (Word-level precision).
* Cut the video into segments (e.g., 5-7 seconds).
* Analyze the visual content (**Qwen2-VL**).
* (Optional) Analyze the background audio (**Qwen2-Audio**).
* Save pairs of `.mp4` and `.txt` files automatically.

### 2. 📝 Bulk Video Captioner
Have a folder full of raw clips? Point the app to it.
* Select **Vision**, **Speech**, or **Both**.
* The app will generate descriptions for every video file in the folder.

### 3. 🖼️ Image Folder Captioner
Standard mode for captioning datasets of images using the powerful Qwen2-VL Vision model.

---

## ⚙️ Installation

### Prerequisites
* **NVIDIA GPU** (Minimum 12GB VRAM recommended for Vision only; 24GB recommended for Audio+Vision mode).
* **Windows 10/11** or **Linux**.

### 🪟 Windows (One-Click)
1.  Run `install.bat`.
    * *This script uses `uv` to create an isolated, conflict-free environment and installs all dependencies including FFmpeg and Flash Attention.*
2.  Run `Run.bat` to start the app.

### 🐧 Linux / WSL
1.  Run `./install.sh`.
2.  Run `./run.sh`.

---

## ⚠️ Important Notes

* **VRAM Usage:** Enabling "Audio Analysis" downloads an additional ~15GB model (Qwen2-Audio). The process will be slower as models are swapped in and out of GPU memory.
* **Models:** The app automatically downloads models to the `./models` folder.
* **RTX 5090 Support:** Includes patches for Blackwell architecture compatibility.

---

## 🏆 Credits

* **[Cyberbol](https://github.com/cyberbol):** Original Creator & Logic Architect.
* **[FNGarvin](https://github.com/FNGarvin):** Engine Architect (UV & Linux Systems).
* **[WildSpeaker7315](https://www.reddit.com/user/WildSpeaker7315/):** Hardware Research & Fixes.

---
<div align="center">
  <b>Licensed under MIT - Built for the Community</b>
</div>