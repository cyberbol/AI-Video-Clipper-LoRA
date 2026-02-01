Major Stability Update: Rewritten installer logic to prevent "Dependency Hell" with WhisperX.

Future-Proof: Added native support for RTX 5090 (Blackwell) via PyTorch Nightly & CUDA 13.0.

Upgraded Core: Now using PyTorch 2.8.0 and CUDA 12.6 for standard cards (4090/3090).

Clean Install: No more red errors during installation!



# 👁️🐧 AI Video Clipper & LoRA Captioner

**Created by: Cyberbol**

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
**⚠️ IMPORTANT:** Before running the installer, you **MUST** have these three tools installed on your Windows system:

1.  **[Python 3.10](https://www.python.org/downloads/release/python-31011/)**
    * *Make sure to check "Add Python to PATH" during installation.*
2.  **[Git](https://git-scm.com/downloads)**
    * *Standard installation.*
3.  **[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)**
    * *Download and run the installer.*
    * *Select the workload: **"Desktop development with C++"**.*
    * *(This is required to compile the WhisperX engine).*

---

## 🚨 IMPORTANT NOTE ABOUT RED ERRORS 🚨
**PLEASE READ THIS:** During the installation (specifically at **Step 4**), you will see **RED ERROR TEXT** in the console saying something like:
> *"ERROR: pip's dependency resolver... conflict... incompatible version..."*

**✅ THIS IS NORMAL AND EXPECTED.**

**Why?** The WhisperX library tries to overwrite our high-performance GPU engine with an older version. 
**Don't worry:** Our smart installer detects this and **AUTOMATICALLY REPAIRS IT** in **Step 5** (Auto-Repair). 
Just ignore the red text and let the script finish.

---

## 🚀 Installation

1.  Open the folder containing these files.
2.  Double-click **`2. Install.bat`**.
3.  Wait for the process to finish. 
    * The script will create an isolated environment.
    * It will install FFmpeg codecs.
    * It will perform the **Auto-Repair**.

> **Note:** If the window closes immediately without doing anything, ensure you have Python installed and added to PATH.

---

## ▶️ How to Run

1.  Double-click **`3. Run.bat`**.
2.  A web interface will open in your default browser.
3.  **Select Your Model:** On the left sidebar, choose between **7B (Quality)** or **2B (Speed)**.
4.  **Upload a Video** (MP4/MKV).
5.  **Configure Settings:**
    * **Target Duration:** Length of clips (e.g., 5-10 seconds).
    * **Trigger Word:** (Optional) If training a LoRA for a specific person, enter their token here (e.g., `prstxxx woman`).
6.  Click **`START PROCESS 🚀`**.

> **⏳ First Run Warning:** > When you select a model for the first time (e.g., you select 7B), the program will download it. 
> * **7B Model:** ~15 GB download.
> * **2B Model:** ~5 GB download.
> * **Whisper Model:** ~3 GB download.
> Please be patient. Future runs will be instant.

---

## 🛠️ Advanced: How to Customize Captions

If you want to change the style of the AI descriptions (e.g., make them shorter, remove audio descriptions, or change the prompt structure), you can edit the source code.

1.  Open **`app.py`** with a text editor (Notepad, VS Code, etc.).
2.  Search for the function: 
    `def generate_vision_caption`
3.  Inside that function, look for the `system_instruction` variable (text inside quotes).
4.  Edit the instructions to fit your needs.

---

## 📂 Output

* All generated clips (`.mp4`) and captions (`.txt`) are saved in the **`dataset`** folder.
* Files are named sequentially (e.g., `001.mp4`, `001.txt`).

---

<div align="center">
  <b>Created by: Cyberbol</b>
</div>

