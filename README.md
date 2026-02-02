# 👁️🐧 AI Video Clipper & LoRA Captioner (v3.2)

**The ultimate automated dataset creation tool.** *Auto-slicing, Transcribing, and Vision Captioning for LoRA/LyCORIS training.*

---

### 🏆 ACKNOWLEDGEMENTS & CREDITS
**Core Engine & RTX 5090 Fixes by [u/WildSpeaker7315](https://www.reddit.com/user/WildSpeaker7315/)** Massive thanks for rewriting the core compatibility layer to support PyTorch 2.6+ and the Blackwell architecture.

**Logic & Workflow by Cyberbol** Strict filtering logic, dynamic folders, UI design, and dataset structuring.

---

### ⚡ Key Features (v3.2)

#### 1. 🎥 Video Auto-Clipper (Strict Mode)
* **Zero-Hallucination Cutting:** The tool uses a "Strict Filter" logic. It searches for sentences that **exactly** match your duration criteria (e.g., 5s with +/- 0.5s tolerance). If a sentence is too long or too short, it is skipped to ensure dataset quality.
* **Dynamic Folders:** Automatically creates folders like `dataset_5.0s`, `dataset_10.0s` so you never overwrite your work.
* **Rich Metadata:** Generates `.txt` files containing: `[Trigger], [Vision Description] Speech: "[Audio Transcript]"`.

#### 2. 🖼️ Image Folder Captioner
* **Batch Processing:** Point to a folder of images, and the AI will caption all of them using Qwen2-VL.
* **Folder Picker:** Native Windows folder selection dialog.

#### 3. ⚙️ Advanced Control
* **RTX 5090 Ready:** Full support for CUDA 13.0 and Blackwell architecture.
* **Custom Vision Prompts:** (New in v3.2) You can now edit the instructions sent to the Vision AI directly in the sidebar (e.g., to focus on camera movement for LTX-Video).
* **Safety Net:** Automatically forces your trigger word into captions if the AI forgets it.

---

### 🛠️ Installation

1.  **Download the latest Release ZIP.**
2.  Extract the folder.
3.  Run **`Install.bat`**.
    * Select **Option [1]** for **RTX 4090 / 3090** (Stable).
    * Select **Option [2]** for **RTX 5090** (Experimental/Nightly).
4.  Once finished, double-click **`Run.bat`** to launch the App.

### ⚙️ Requirements
* Windows 10/11
* NVIDIA GPU (8GB+ VRAM recommended, 24GB for best performance)
* [Python 3.10](https://www.python.org/downloads/) installed (Add to PATH)
* [Git](https://git-scm.com/downloads) installed
* [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Desktop development with C++)

### 🤝 Contributing
This is a community-driven project born from personal needs. I am not a pro developer. If you find bugs or want to improve the logic, feel free to fork or reach out on Reddit!

---
*v3.2 Update: Optimized default prompts for LTX-Video & Wan2.1 training.*