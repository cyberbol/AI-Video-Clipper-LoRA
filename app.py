# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner (v3.7 - UI Flow Fix)
# 🏆 CREDITS: Cyberbol (Logic), FNGarvin (Engine), WildSpeaker (5090 Fix)
# --------------------------------------------------------------------------------

import os
import sys
import streamlit as st
import whisperx
from moviepy import VideoFileClip
import tempfile
import torch
import gc
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import tkinter as tk
from tkinter import filedialog

# --- 1. BOOTSTRAP & PATCHES ---
try:
    import patches
    patches.apply_patches()
except ImportError:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import torch.serialization
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata, Node
        import typing
        torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, Node, typing.Any])
    except: pass
    if not hasattr(torch, "_patched_for_5090"):
        _orig_load = torch.load
        def _safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        torch.load = _safe_load
        torch._patched_for_5090 = True

# --- 2. KONFIGURACJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# --- 3. UI CONFIG ---
st.set_page_config(page_title="AI Clipper v3.7 (Ultimate)", layout="wide")
st.title("👁️🐧 AI Video Clipper & LoRA Captioner")
st.markdown("v3.7 | Created by: **Cyberbol** | Engine: **FNGarvin** (UV) | 5090 Fix: **WildSpeaker**")

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

st.sidebar.divider()
model_choice = st.sidebar.radio("Vision Model:", ["Qwen2-VL-7B (High Quality)", "Qwen2-VL-2B (Fast)"], index=0)
SELECTED_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct" if "7B" in model_choice else "Qwen/Qwen2-VL-2B-Instruct"

st.sidebar.divider()
st.sidebar.markdown("### 📝 Vision Instructions")
default_prompt = "Describe this {type} in detail for a dataset. Main subject: {trigger}. Describe the action, camera movement, lighting, atmosphere, and background."
user_instruction = st.sidebar.text_area("System Prompt:", value=default_prompt, height=150)

# --- 4. FUNKCJE ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def load_vision_models():
    st.info(f"⏳ Loading Vision Engine ({SELECTED_MODEL_ID})...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        SELECTED_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa", low_cpu_mem_usage=True 
    )
    processor = AutoProcessor.from_pretrained(SELECTED_MODEL_ID)
    return model, processor

def select_folder_dialog():
    root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root); root.destroy()
    return folder_path

def caption_content(content_path, content_type, model, processor, trigger, custom_prompt, model_id):
    if not custom_prompt.strip(): custom_prompt = "Describe this content in high detail."
    if "2B" in model_id: custom_prompt += " Output as a single, continuous paragraph. No markdown."
    final_prompt = custom_prompt.replace("{trigger}", trigger if trigger else "").replace("{type}", "video" if content_type == "video" else "image")
    messages = [{"role": "user", "content": [{"type": content_type, content_type: content_path, "max_pixels": 360*420, "fps": 1.0}, {"type": "text", "text": final_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    output_text = output_text.replace("**", "").replace("##", "")
    if trigger and trigger not in output_text: output_text = f"{trigger}, {output_text}"
    return output_text

# --- 5. LOGIKA UI ---
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

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
    
    if uploaded_file and st.button("🚀 START PROCESSING"):
        status_box = st.empty()
        status_box.info("🚀 **Phase 1/2: Initializing Audio Analysis...** WhisperX is starting up.")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read()); video_path = tmp.name
        try:
            status_box.info("⏳ **Phase 1/2: Analyzing Speech & Timestamps...** Finding the best clips.")
            model_w = whisperx.load_model("large-v3", device, compute_type="float16", download_root=MODELS_DIR)
            audio = whisperx.load_audio(video_path); result = model_w.transcribe(audio, batch_size=16)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)
            del model_w, model_a; clear_vram()
            
            segments = [s for s in result["segments"] if (target_dur - tol_minus) <= (s['end'] - s['start']) <= (target_dur + tol_plus)]
            
            if segments:
                status_box.empty()
                folder_name = project_name.strip() if project_name.strip() else f"dataset_{target_dur}s"
                out_dir = os.path.join(BASE_DIR, folder_name); os.makedirs(out_dir, exist_ok=True)
                v_model, v_proc = load_vision_models(); video_f = VideoFileClip(video_path)
                st.success(f"Found {len(segments)} clips. Saving to: {out_dir}")
                prog = st.progress(0)
                for i, seg in enumerate(segments[:100]):
                    base = f"clip_{i+1:03d}"; c_path = os.path.join(out_dir, f"{base}.mp4")
                    sub = video_f.subclipped(seg['start'], seg['end'])
                    if not keep_orig: sub = sub.resized(new_size=(out_width, out_height)).write_videofile(c_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                    else: sub.write_videofile(c_path, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                    cap = caption_content(c_path, "video", v_model, v_proc, lora_trigger, user_instruction, SELECTED_MODEL_ID)
                    speech = seg['text'].strip()
                    with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f: f.write(f"{cap} The person says: \"{speech}\"")
                    with st.expander(f"✅ {base}"):
                        st.video(c_path); st.info(f"💬 **Speech:** {speech}")
                    prog.progress((i+1)/len(segments))
                video_f.close()
                st.success("✅ DONE! Processing finished successfully.") # DODANO: Napis DONE
            else:
                st.warning("No segments found matching those exact duration margins.")
                status_box.empty()
        finally:
            clear_vram(); 
            if os.path.exists(video_path): os.unlink(video_path)

elif app_mode == "📝 Bulk Video Captioner":
    if 'v_bulk_path' not in st.session_state: st.session_state['v_bulk_path'] = ""
    col_v, col_vbtn = st.columns([3, 1])
    with col_vbtn:
        if st.button("📂 Select Folder"): # NAPRAWIONE: Teraz tylko po kliknięciu
            sel = select_folder_dialog()
            if sel: st.session_state['v_bulk_path'] = sel; st.rerun()
    with col_v: v_dir = st.text_input("Folder Path:", value=st.session_state['v_bulk_path'])
    if st.button("🚀 START BULK CAPTIONING") and os.path.exists(v_dir):
        v_model, v_proc = load_vision_models()
        videos = [f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".mkv"))]
        prog = st.progress(0)
        for i, v_name in enumerate(videos):
            p = os.path.join(v_dir, v_name); cap = caption_content(p, "video", v_model, v_proc, lora_trigger, user_instruction, SELECTED_MODEL_ID)
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: f.write(cap)
            prog.progress((i+1)/len(videos))
        st.success("✅ DONE! Bulk Captioning finished.") # DODANO: Napis DONE
        clear_vram()

else: # IMAGE CAPTIONER
    if 'img_path' not in st.session_state: st.session_state['img_path'] = ""
    col_p, col_b = st.columns([3, 1])
    with col_b:
        if st.button("📂 Select Folder"): # NAPRAWIONE: Teraz tylko po kliknięciu
            sel = select_folder_dialog()
            if sel: st.session_state['img_path'] = sel; st.rerun()
    with col_p: img_dir = st.text_input("Path:", value=st.session_state['img_path'])
    if st.button("🚀 CAPTION FOLDER") and os.path.exists(img_dir):
        v_model, v_proc = load_vision_models()
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".webp"))]
        prog = st.progress(0)
        for i, name in enumerate(imgs):
            p = os.path.join(img_dir, name); cap = caption_content(p, "image", v_model, v_proc, lora_trigger, user_instruction, SELECTED_MODEL_ID)
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: f.write(cap)
            prog.progress((i+1)/len(imgs))
        st.success("✅ DONE! Folder Captioning finished.") # DODANO: Napis DONE
        clear_vram()

st.markdown("---")
st.markdown("<center><b>Project maintained by Cyberbol</b></center>", unsafe_allow_html=True)