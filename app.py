# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner
# 🏆 CREDITS: Cyberbol (Logic), FNGarvin (Engine), WildSpeaker (5090 Fix)
# --------------------------------------------------------------------------------

import os
import sys

# --- 1. BOOTSTRAP & PATCHES ---
# 🚨 CRITICAL: Patches must apply BEFORE heavy imports (Torch, WhisperX)
try:
    import patches
    patches.apply_patches()
except ImportError:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Basic fallbacks if patches.py is missing
    pass

import streamlit as st
import whisperx
from moviepy import VideoFileClip
import tempfile
import torch
import gc
import time
import tkinter as tk
from tkinter import filedialog
from downloader import download_model
import logging

# Suppress Streamlit's "missing ScriptRunContext" warning in threads
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# 5090 Fix (In-line backup if patches failed or specifically for this file)
if not hasattr(torch, "_patched_for_5090"):
    _orig_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _safe_load
    torch._patched_for_5090 = True

# --- 2. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# --- 3. UI CONFIG ---
st.set_page_config(page_title="AI Clipper", layout="wide")
st.title("👁️🐧 AI Video Clipper & LoRA Captioner")
st.markdown("Created by: **Cyberbol** | Engine: **FNGarvin** (Unified) | 5090 Fix: **WildSpeaker**")

device = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.header("⚙️ Engine Status")
if device == "cuda":
    st.sidebar.success(f"GPU: **{torch.cuda.get_device_name(0)}**")
else:
    st.sidebar.error("CUDA not detected!")

st.sidebar.divider()
app_mode = st.sidebar.selectbox("Choose Mode:", [
    "🎥 Video Auto-Clipper", 
    "📝 Bulk Video Captioner",
    "🖼️ Image Folder Captioner"
])

# Import Unified Engine & Scanner
from vision_engine import VisionEngine, scan_local_gguf_models

st.sidebar.divider()
# --- MODEL SELECTION (Radio Buttons) ---
model_options = {
    "GGUF: Gemma-3-12B (Next-Gen, 4-bit)": {
        "backend": "gguf",
        "repo": "unsloth/gemma-3-12b-it-GGUF",
        "model": "gemma-3-12b-it-IQ4_XS.gguf",
        "projector": "mmproj-F16.gguf"
    },
    "GGUF: Qwen3-VL-8B-Instruct (Q4_K_M)": {
        "backend": "gguf",
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "model": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "projector": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
    },
    "Transformer: Qwen2-VL-7B (Legacy)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-7B-Instruct"
    },
    "Transformer: Qwen2-VL-2B (Legacy)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-2B-Instruct"
    }
}

# Auto-Discovery
local_ggufs = scan_local_gguf_models(MODELS_DIR)
# Filter duplicates
existing_ggufs = [m["model"] for m in model_options.values() if m.get("backend") == "gguf"]
for label, config in local_ggufs.items():
    if config["model"] not in existing_ggufs:
        model_options[label] = config

# Default to Transformer: Qwen2-VL-7B (Legacy) which is index 2 in the dict above
# Dict ordering is preserved in modern Python. 
model_label = st.sidebar.radio("Vision Model:", list(model_options.keys()), index=2)
SELECTED_MODEL = model_options[model_label]

st.sidebar.divider()
st.sidebar.markdown("### 📝 Vision Instructions")
default_prompt = "Describe this {type} in detail for a dataset. Main subject: {trigger}. Describe the action, camera movement, lighting, atmosphere, and background."
user_instruction = st.sidebar.text_area("System Prompt:", value=default_prompt, height=150)
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

# --- SIDEBAR: ADVANCED OPTIONS ---
with st.sidebar.expander("🛠️ Advanced Generation Options"):
    gen_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    gen_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    gen_max_tokens = st.number_input("Max New Tokens", 64, 2048, 256)
    
GEN_CONFIG = {
    "temperature": gen_temp,
    "top_p": gen_top_p,
    "max_tokens": gen_max_tokens
}

# --- 4. HELPERS ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def load_vision_engine():
    """Unified loading using VisionEngine class"""
    # Check if model changed
    if 'vision_engine' not in st.session_state or st.session_state.get('last_model_config') != str(SELECTED_MODEL):
        
        # Clear old
        if 'vision_engine' in st.session_state and st.session_state['vision_engine']:
            st.session_state['vision_engine'].clear()
            
        with st.status(f"🚀 Initializing Vision Engine...", expanded=True) as status:
            engine = VisionEngine(SELECTED_MODEL, device=device, models_dir=MODELS_DIR)
            engine.load(log_callback=status.write)
            status.update(label="✅ Vision Engine Ready!", state="complete", expanded=False)
            
        st.session_state['vision_engine'] = engine
        st.session_state['last_model_config'] = str(SELECTED_MODEL)
    
    return st.session_state['vision_engine']

def select_folder_dialog():
    root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root); root.destroy()
    return folder_path

# --- 5. APP LOGIC ---

if app_mode == "🎥 Video Auto-Clipper":
    project_name = st.text_input("Project Name (Optional)", value="")
    uploaded_file = st.file_uploader("Upload Video (MP4, MKV)", type=["mp4", "mkv"])
    
    st.subheader("✂️ Cutting Parameters")
    keep_orig = st.checkbox("Keep Original Resolution & FPS", value=False)
    col1, col2, col3, col4 = st.columns(4)
    with col1: target_dur = st.number_input("Target Length (s)", 1.0, 60.0, 5.0)
    with col2: out_width = st.number_input("Output Width", 256, 3840, 1024, disabled=keep_orig)
    with col3: out_height = st.number_input("Output Height", 256, 3840, 1024, disabled=keep_orig)
    with col4: out_fps = st.number_input("Output FPS", 1, 120, 24, disabled=keep_orig)
    
    t_col1, t_col2 = st.columns(2)
    with t_col1: tol_minus = st.number_input("Tolerance Margin - (sec)", 0.0, 5.0, 0.0)
    with t_col2: tol_plus = st.number_input("Tolerance Margin + (sec)", 0.0, 10.0, 0.5)
    
    col_btn, col_timer = st.columns([1, 4])
    with col_btn: start_processing = st.button("🚀 START PROCESSING")
    with col_timer: timer_placeholder = st.empty()

    if uploaded_file and start_processing:
        start_ts = time.time()
        timer_placeholder.info("⏱️ Processing started...")
        status_box = st.empty()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read()); video_path = tmp.name
            
        try:
            # --- PHASE 1: WHISPER (Robust Loading) ---
            status_box.info("🚀 Phase 1: Audio Analysis (WhisperX)...")
            
            # Use downloader to ensure model is present and get local path
            # This bypasses huggingface_hub's strict offline checks that fail
            wx_repo = "Systran/faster-whisper-large-v3"
            with st.spinner("Ensuring Audio Model..."):
                wx_path = download_model(wx_repo, MODELS_DIR, log_callback=None)
            
            model_w = whisperx.load_model(wx_path, device, compute_type="float16")
            audio = whisperx.load_audio(video_path)
            result = model_w.transcribe(audio, batch_size=16)
            
            # Alignment (Fix: Explicit directory to avoid redownload loop/hub issues)
            align_model_dir = os.path.join(MODELS_DIR, "PyTorch")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], 
                device=device,
                model_dir=align_model_dir
            )
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)

            
            # Cleanup Audio Resources
            del model_w, model_a, audio, metadata
            clear_vram()
            
            # Filter Segments
            segments = [s for s in result["segments"] if (target_dur - tol_minus) <= (s['end'] - s['start']) <= (target_dur + tol_plus)]
            
            if segments:
                status_box.empty()
                folder_name = project_name.strip() if project_name.strip() else f"dataset_{target_dur}s"
                out_dir = os.path.join(BASE_DIR, folder_name); os.makedirs(out_dir, exist_ok=True)
                
                # Load Vision Engine
                v_engine = load_vision_engine()
                video_f = VideoFileClip(video_path)
                
                st.success(f"Found {len(segments)} clips. Saving to: {out_dir}")
                prog = st.progress(0)
                
                for i, seg in enumerate(segments[:100]):
                    base = f"clip_{i+1:03d}"
                    c_path = os.path.join(out_dir, f"{base}.mp4")
                    
                    # Create Clip
                    sub = video_f.subclipped(seg['start'], seg['end'])
                    if not keep_orig:
                        sub = sub.resized(new_size=(out_width, out_height))
                        sub.write_videofile(c_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                    else:
                        sub.write_videofile(c_path, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                    
                    # Captioning With Streaming
                    stream_box = st.empty()
                    def on_token(text):
                        stream_box.markdown(f"**📝 Generuję:** {text}")
                    
                    cap = v_engine.caption(c_path, "video", lora_trigger, user_instruction, 
                                           gen_config=GEN_CONFIG, stream_callback=on_token)
                    stream_box.empty() # Clear transient stream
                    
                    speech = seg['text'].strip()
                    final_txt = f"{cap} The person says: \"{speech}\""
                    
                    with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f: 
                        f.write(final_txt)
                        
                    with st.expander(f"✅ {base}"):
                        st.video(c_path)
                        st.info(f"**Final:** {final_txt}")
                        
                    prog.progress((i+1)/len(segments))
                
                video_f.close()
                st.success("✅ DONE! Processing finished.")
                
                # Timing
                total_seconds = time.time() - start_ts
                mins, secs = divmod(total_seconds, 60)
                timer_placeholder.success(f"⏱️ Total Time: {int(mins)}m {int(secs)}s")
            else:
                st.warning("No segments match restrictions.")
                
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        finally:
            clear_vram()
            if os.path.exists(video_path): os.unlink(video_path)

elif app_mode == "📝 Bulk Video Captioner":
    if 'v_bulk_path' not in st.session_state: st.session_state['v_bulk_path'] = ""
    col_v, col_vbtn = st.columns([3, 1])
    with col_vbtn:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['v_bulk_path'] = sel; st.rerun()
    with col_v: v_dir = st.text_input("Folder Path:", value=st.session_state['v_bulk_path'])
    
    if st.button("🚀 START BULK CAPTIONING") and os.path.exists(v_dir):
        start_ts = time.time()
        v_engine = load_vision_engine()
        videos = [f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".mkv"))]
        
        prog = st.progress(0)
        status_text = st.empty()
        
        for i, v_name in enumerate(videos):
            p = os.path.join(v_dir, v_name)
            
            stream_box = st.empty()
            def on_token(text):
                stream_box.markdown(f"**Processing {v_name}:** {text}")
                
            cap = v_engine.caption(p, "video", lora_trigger, user_instruction, 
                                   gen_config=GEN_CONFIG, stream_callback=on_token)
            stream_box.empty()
            
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: 
                f.write(cap)
            prog.progress((i+1)/len(videos))
            
        st.success("✅ DONE! Bulk Captioning finished.")

else: # IMAGE CAPTIONER
    if 'img_path' not in st.session_state: st.session_state['img_path'] = ""
    col_p, col_b = st.columns([3, 1])
    with col_b:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['img_path'] = sel; st.rerun()
    with col_p: img_dir = st.text_input("Path:", value=st.session_state['img_path'])
    
    if st.button("🚀 CAPTION FOLDER") and os.path.exists(img_dir):
        start_ts = time.time()
        v_engine = load_vision_engine()
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".webp"))]
        
        prog = st.progress(0)
        for i, name in enumerate(imgs):
            p = os.path.join(img_dir, name)
            
            stream_box = st.empty()
            def on_token(text):
                stream_box.markdown(f"**{name}:** {text}")
                
            cap = v_engine.caption(p, "image", lora_trigger, user_instruction, 
                                   gen_config=GEN_CONFIG, stream_callback=on_token)
            stream_box.empty()
            
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: 
                f.write(cap)
            prog.progress((i+1)/len(imgs))
            
        st.success("✅ DONE! Folder finished.")

st.markdown("---")
st.markdown("<center><b>Project maintained by Cyberbol</b></center>", unsafe_allow_html=True)