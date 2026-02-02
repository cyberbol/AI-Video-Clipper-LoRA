# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner (v3.6 Ultimate - Native Quality Update)
#
# 🏆 CREDITS & CONTRIBUTORS:
# - Cyberbol:        Original Creator, Logic Design, Strict Mode, Custom Prompts
# - FNGarvin:        System Architect, UV Engine, Linux/WSL Support
# - WildSpeaker7315: Blackwell (RTX 5090) Compatibility Research
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
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset") 
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# --- 3. UI CONFIG ---
st.set_page_config(page_title="AI Clipper v3.6 (Native Quality)", layout="wide")
st.title("👁️🐧 AI Video Clipper & LoRA Captioner")
st.markdown("v3.6 | Created by: **Cyberbol** | Engine: **FNGarvin** (UV) | 5090 Fix: **WildSpeaker**")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SIDEBAR: GPU INFO ---
st.sidebar.header("⚙️ Engine Status")
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"GPU: **{gpu_name}**")
else:
    st.sidebar.error("CUDA not detected!")

st.sidebar.divider()
app_mode = st.sidebar.selectbox("Choose Mode:", ["🎥 Video Auto-Clipper", "🖼️ Image Folder Captioner"])

st.sidebar.divider()
model_choice = st.sidebar.radio("Vision Model:", ["Qwen2-VL-7B (High Quality)", "Qwen2-VL-2B (Fast)"], index=0)
SELECTED_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct" if "7B" in model_choice else "Qwen/Qwen2-VL-2B-Instruct"

st.sidebar.divider()
st.sidebar.markdown("### 📝 Vision Instructions")
default_prompt = "Describe this {type} in detail for a dataset. Main subject: {trigger}. Describe the action, camera movement, lighting, atmosphere, and background."
user_instruction = st.sidebar.text_area("System Prompt:", value=default_prompt, height=150)
st.sidebar.info(f"💡 **Tip:** Use `{{trigger}}` tag to automatically insert your word.")

# --- 4. FUNKCJE POMOCNICZE ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def load_vision_models():
    st.info(f"⏳ Loading Vision Engine ({SELECTED_MODEL_ID})...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        SELECTED_MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True 
    )
    processor = AutoProcessor.from_pretrained(SELECTED_MODEL_ID)
    return model, processor

def select_folder_dialog():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

# --- 5. LOGIKA WIDEO (STRICT MODE) ---
def run_whisper_phase(video_path, target_dur, tol_min, tol_max):
    st.info("⏳ 1/3 Audio Analysis (Strict Filtering)...")
    model = whisperx.load_model("large-v3", device, compute_type="float16", download_root=MODELS_DIR)
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=16)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    del model, model_a
    clear_vram()
    final_clips = []
    min_accept = target_dur - tol_min
    max_accept = target_dur + tol_max
    if min_accept < 0.1: min_accept = 0.1
    for s in result["segments"]:
        duration = s['end'] - s['start']
        if min_accept <= duration <= max_accept:
            final_clips.append(s)
    return final_clips

def caption_content(content_path, content_type, model, processor, trigger, custom_prompt, model_id):
    if not custom_prompt.strip():
        custom_prompt = "Describe this content in high detail."
    if "2B" in model_id:
        custom_prompt += " Output as a single, continuous paragraph. No markdown."
    final_prompt = custom_prompt.replace("{trigger}", trigger if trigger else "").replace("{type}", "video" if content_type == "video" else "image")
    messages = [{"role": "user", "content": [{"type": content_type, content_type: content_path, "max_pixels": 360*420, "fps": 1.0}, {"type": "text", "text": final_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    output_text = output_text.replace("**", "").replace("##", "")
    if trigger and trigger not in output_text:
        output_text = f"{trigger}, {output_text}"
    return output_text

# --- 6. UI GŁÓWNE ---
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

if app_mode == "🎥 Video Auto-Clipper":
    uploaded_file = st.file_uploader("Upload Video (MP4, MKV)", type=["mp4", "mkv"])
    st.subheader("✂️ Cutting Parameters")
    
    # NOWOŚĆ v3.6: Checkbox dla natywnej jakości
    keep_orig = st.checkbox("Keep Original Resolution & FPS", value=False, help="If checked, clips will have the same resolution and framerate as the source.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: target_dur = st.number_input("Target Length (s)", 1.0, 60.0, 5.0)
    with col2: out_width = st.number_input("Output Width (px)", 256, 3840, 1024, disabled=keep_orig)
    with col3: out_height = st.number_input("Output Height (px)", 256, 3840, 1024, disabled=keep_orig)
    with col4: out_fps = st.number_input("Output FPS", 1, 120, 24, disabled=keep_orig)
    
    t_col1, t_col2 = st.columns(2)
    with t_col1: tol_minus = st.number_input("Tolerance Margin - (sec)", 0.0, 5.0, 0.0)
    with t_col2: tol_plus = st.number_input("Tolerance Margin + (sec)", 0.0, 10.0, 0.5)

    if uploaded_file and st.button("🚀 START PROCESSING"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        try:
            segments = run_whisper_phase(video_path, target_dur, tol_minus, tol_plus)
            if not segments:
                st.error("No clips found in this duration range!")
            else:
                current_output_dir = os.path.join(BASE_DIR, f"dataset_{target_dur}s")
                os.makedirs(current_output_dir, exist_ok=True)
                
                v_model, v_processor = load_vision_models()
                video_full = VideoFileClip(video_path)
                
                # --- PRZYWRÓCONE INFO O LICZBIE KLIPÓW ---
                st.success(f"Found {len(segments)} PERFECT clips. Saving to: {current_output_dir}")
                prog = st.progress(0)
                
                for i, seg in enumerate(segments[:100]):
                    base_name = f"clip_{i+1:03d}"
                    clip_out = os.path.join(current_output_dir, f"{base_name}.mp4")
                    sub = video_full.subclipped(seg['start'], seg['end'])
                    
                    # LOGIKA v3.6: Resizing lub Native Quality
                    if not keep_orig:
                        sub = sub.resized(new_size=(out_width, out_height))
                        sub.write_videofile(clip_out, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                    else:
                        sub.write_videofile(clip_out, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                    
                    visual_caption = caption_content(clip_out, "video", v_model, v_processor, lora_trigger, user_instruction, SELECTED_MODEL_ID)
                    
                    # Zapis opisu
                    with open(os.path.join(current_output_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"{visual_caption} The person says: \"{seg['text'].strip()}\"")
                    
                    # --- PRZYWRÓCONY PODGLĄD W UI ---
                    with st.expander(f"✅ Clip {i+1} ({seg['end']-seg['start']:.1f}s)"):
                        st.video(clip_out)
                        st.write(f"**Speech:** {seg['text']}")
                    
                    prog.progress((i+1)/min(len(segments), 100))
                
                video_full.close()
                st.success(f"Processing Complete! Dataset ready in: {current_output_dir}")
                
        except Exception as e:
            st.error(f"Error during processing: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            clear_vram()
            if os.path.exists(video_path): os.unlink(video_path)

else:
    # IMAGE MODE (v3.5 logic)
    if 'img_folder_path' not in st.session_state: st.session_state['img_folder_path'] = "C:/Dataset/Images"
    col_path, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("📂 Browse Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['img_folder_path'] = sel; st.rerun()
    with col_path: img_folder = st.text_input("Path:", value=st.session_state['img_folder_path'])
    if st.button("🚀 CAPTION FOLDER"):
        if os.path.exists(img_folder):
            v_model, v_processor = load_vision_models()
            images = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            if not images:
                st.warning("No images found!")
            else:
                prog = st.progress(0)
                for i, img_name in enumerate(images):
                    full_path = os.path.join(img_folder, img_name)
                    caption = caption_content(full_path, "image", v_model, v_processor, lora_trigger, user_instruction, SELECTED_MODEL_ID)
                    with open(os.path.join(img_folder, os.path.splitext(img_name)[0] + ".txt"), "w", encoding="utf-8") as f:
                        f.write(caption)
                    st.write(f"✅ Captioned: {img_name}")
                    prog.progress((i+1)/len(images))
                st.success("Folder Captioning Complete!")
                clear_vram()

st.markdown("---")
st.markdown("<center><b>Project maintained by Cyberbol</b> | v3.6 Native Quality Update</center>", unsafe_allow_html=True)