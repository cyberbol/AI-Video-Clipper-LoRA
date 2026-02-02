# 👁️🐧 AI Video Clipper & LoRA Captioner (v3.5)

![Demo](demo.webp)

**The ultimate automated dataset creation tool.**
*Auto-slicing, Transcribing, and Vision Captioning for LoRA/LyCORIS training.*

---

### 🏆 PROJECT CREDITS
This project is a collaborative effort driven by the open-source community:

* **[Cyberbol](https://github.com/cyberbol):** **Original Creator & Logic.** Designed the UI, "Strict Mode" filtering, dynamic folder logic, and Vision Prompt engineering.
* **[FNGarvin](https://github.com/FNGarvin):** **System Architect.** Modernized the engine using `uv`, implemented Linux/WSL support, optimized privacy (headless mode), and cleaned up the code architecture.
* **[WildSpeaker7315](https://www.reddit.com/user/WildSpeaker7315/):** **Blackwell Research.** Provided the critical breakdown of PyTorch/CUDA compatibility for RTX 5090 support.

---

### ⚡ Key Features (v3.5 Ultimate)

#### 1. 🚀 Next-Gen Engine (Powered by `uv`)
* **Instant Setup:** Uses **`uv`** package manager. Installation takes seconds, not hours.
* **Disk Saver:** Uses hardlinks to minimize disk usage (no more 10GB duplicated venvs).
* **Cross-Platform:** Native support for **Windows**, **Linux**, and **WSL**.
* **RTX 5090 Ready:** Native support for CUDA 12.8 / PyTorch 2.10.

#### 2. 🎥 Intelligent Auto-Clipper (Strict Mode)
* **Zero-Hallucination Cutting:** The tool uses a "Strict Filter" logic. It searches for sentences that **exactly** match your duration criteria (e.g., 5s with +/- 0.5s tolerance).
* **Dynamic Folders:** Automatically creates folders like `dataset_5.0s`, `dataset_10.0s` to prevent overwriting.

#### 3. 🖼️ Vision Captioning (Qwen2-VL)
* **Two Modes:** Switch between **7B (High Quality)** and **2B (Speed)** instantly.
* **Custom Prompts:** You can edit the Vision AI instructions in the sidebar (Perfect for **LTX-Video** camera movement descriptions).
* **Safety Net:** Automatically forces your Trigger Word into the caption if the AI forgets it.

---

## 🚀 Installation

### 🪟 Windows
1.  Double-click **`install.bat`**.
    * *(Note: The installer will automatically download Python and dependencies via `uv`. No manual setup required.)*
2.  Once finished, double-click **`Run.bat`** to start the app.

### 🐧 Linux / WSL
1.  Open a terminal in the project folder.
2.  Make scripts executable:
    ```bash
    chmod +x install.sh run.sh
    ```
3.  Run the installer:
    ```bash
    ./install.sh
    ```
4.  Start the app:
    ```bash
    ./run.sh
    ```

---

## ⚙️ How to Use
1.  **Select Mode:** Choose between *Video Auto-Clipper* or *Image Folder Captioner*.
2.  **Choose Model:** Select **7B** (Quality) or **2B** (Speed) in the sidebar.
3.  **Custom Prompt (Optional):** In the sidebar, you can define how the AI should describe the scene (e.g., *"Focus on lighting and camera angle"*).
4.  **Upload Video / Select Folder.**
5.  **Click START.**

---

<div align="center">
  <b>Open Source Community Project</b>
</div>