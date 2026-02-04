# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner (v4.0 Stable)
# 🏆 CREDITS: Cyberbol (Logic), FNGarvin (Engine), WildSpeaker (5090 Fix)
# --------------------------------------------------------------------------------

import os
import sys
import streamlit as st
import whisperx
from moviepy import VideoFileClip, AudioFileClip
import tempfile
import torch
import gc
import time
import librosa
import numpy as np
import re
from transformers import Qwen2VLForConditionalGeneration, Qwen2AudioForConditionalGeneration, AutoProcessor
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
st.set_page_config(page_title="AI Clipper v4.0", layout="wide")
st.title("👁️🐧👂 AI Video Clipper & LoRA Captioner")
st.markdown("v4.0 Stable | **Cyberbol** (Logic) | **FNGarvin** (Engine) | **WildSpeaker** (Fixes)")

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
# Vision Model Selection
model_choice = st.sidebar.radio("Vision Model:", ["Qwen2-VL-7B (High Quality)", "Qwen2-VL-2B (Fast)"], index=0)
SELECTED_VISION_ID = "Qwen/Qwen2-VL-7B-Instruct" if "7B" in model_choice else "Qwen/Qwen2-VL-2B-Instruct"
# Audio Model ID
SELECTED_AUDIO_ID = "Qwen/Qwen2-Audio-7B-Instruct"

st.sidebar.divider()
st.sidebar.markdown("### 📝 Instructions")

# PROMPTY
default_prompt = "Describe this {type} concisely in one fluid paragraph. Focus on the main action, subject, camera movement, and lighting. Do not list details."
user_instruction = st.sidebar.text_area("Vision Prompt:", value=default_prompt, height=100)

audio_prompt_default = "Describe the sound atmosphere and music mood in one short, non-technical sentence. Do not mention BPM or keys."
audio_prompt = st.sidebar.text_area("Audio Prompt (Qwen2-Audio):", value=audio_prompt_default, height=60)

# --- 4. FUNKCJE MEMORY ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- 5. MODEL LOADERS ---
def load_vision_model():
    st.info(f"⏳ Loading Vision Engine ({SELECTED_VISION_ID})...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        SELECTED_VISION_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa", low_cpu_mem_usage=True 
    )
    processor = AutoProcessor.from_pretrained(SELECTED_VISION_ID)
    return model, processor

def load_audio_model():
    st.info(f"⏳ Loading Audio Intelligence Engine ({SELECTED_AUDIO_ID})...")
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            SELECTED_AUDIO_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa", low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(SELECTED_AUDIO_ID)
        return model, processor
    except Exception as e:
        st.error(f"Failed to load Qwen2-Audio: {e}")
        return None, None

def select_folder_dialog():
    root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root); root.destroy()
    return folder_path

# --- 6. CAPTIONING FUNCTIONS ---
def caption_vision(content_path, content_type, model, processor, trigger, custom_prompt, model_id):
    if not custom_prompt.strip(): custom_prompt = "Describe this content concisely."
    if "2B" in model_id: custom_prompt += " Output as a single sentence."
    final_prompt = custom_prompt.replace("{trigger}", trigger if trigger else "").replace("{type}", "video" if content_type == "video" else "image")
    messages = [{"role": "user", "content": [{"type": content_type, content_type: content_path, "max_pixels": 360*420, "fps": 1.0}, {"type": "text", "text": final_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=160)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    output_text = output_text.replace("**", "").replace("##", "").replace("\n", " ").strip()
    if trigger and trigger not in output_text: output_text = f"{trigger}, {output_text}"
    return output_text

def caption_audio(audio_path, model, processor, custom_prompt):
    y, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    conversation = [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {'type': 'text', 'text': custom_prompt}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=y, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=48)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    response = response.replace("\n", " ").strip()
    if response.endswith("."): response = response[:-1]
    return response

# --- 7. LOGIKA UI ---
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

# =================================================================================================
# MODE 1: VIDEO AUTO-CLIPPER (v4.0 Logic with Audio Intelligence)
# =================================================================================================
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
    
    st.markdown("---")
    
    c_ltx, c_audio = st.columns(2)
    with c_ltx:
        enable_hard_cut = st.checkbox("⚡ LTX Hard Cut Mode (Fixed Duration)", value=False, help="Strict cut. Ignores sentence end.")
        if enable_hard_cut:
            max_clips = st.number_input("Max Clips Limit", 1, 500, 20)
            tol_minus, tol_plus = 0, 0
        else:
            tol_minus = st.number_input("Tolerance - (s)", 0.0, 5.0, 0.0)
            tol_plus = st.number_input("Tolerance + (s)", 0.0, 10.0, 0.5)
            
    with c_audio:
        enable_audio_cap = st.checkbox("🎧 Enable Audio Analysis (Qwen2-Audio)", value=False, help="Adds description of background sounds.")

    col_btn, col_timer = st.columns([1, 4])
    with col_btn:
        start_processing = st.button("🚀 START PROCESSING")
    with col_timer:
        timer_placeholder = st.empty()

    if uploaded_file and start_processing:
        start_ts = time.time()
        timer_placeholder.info("⏱️ Processing started...")
        status_box = st.empty()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read()); video_path = tmp.name
        
        try:
            # === PHASE 1: WHISPER X ===
            status_box.info("🚀 **Phase 1/3: Speech Analysis (WhisperX)**...")
            check_clip = VideoFileClip(video_path); video_duration = check_clip.duration; check_clip.close(); del check_clip

            model_w = whisperx.load_model("large-v3", device, compute_type="float16", download_root=MODELS_DIR)
            audio_source = whisperx.load_audio(video_path)
            result = model_w.transcribe(audio_source, batch_size=16)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio_source, device)
            
            all_words_global = []
            for s in result["segments"]:
                if 'words' in s: all_words_global.extend(s['words'])

            del model_w, model_a, audio_source, metadata
            clear_vram()
            
            # === SEGMENT SELECTION ===
            final_segments = []
            if enable_hard_cut:
                current_time_pointer = 0.0
                for s in result["segments"]:
                    if len(final_segments) >= max_clips: break
                    start_t = s['start']
                    if start_t >= current_time_pointer:
                        end_t = start_t + target_dur
                        if end_t <= video_duration:
                            custom_seg = s.copy(); custom_seg['end'] = end_t
                            final_segments.append(custom_seg)
                            current_time_pointer = end_t
            else:
                final_segments = [s for s in result["segments"] if (target_dur - tol_minus) <= (s['end'] - s['start']) <= (target_dur + tol_plus)]

            if not final_segments:
                st.warning("No segments found."); status_box.empty(); st.stop()

            # === PHASE 2: AUDIO CAPTIONING ===
            audio_captions_map = {}
            if enable_audio_cap:
                status_box.info(f"👂 **Phase 2/3: Audio Analysis ({len(final_segments)} clips)**...")
                a_model, a_proc = load_audio_model()
                if a_model:
                    prog_a = st.progress(0)
                    full_audio_clip = AudioFileClip(video_path)
                    for i, seg in enumerate(final_segments[:100]):
                        cut_end = seg['end'] if not enable_hard_cut else (seg['start'] + target_dur)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_aud:
                            sub_a = full_audio_clip.subclipped(seg['start'], cut_end)
                            sub_a.write_audiofile(tmp_aud.name, logger=None)
                            tmp_aud_path = tmp_aud.name
                        try:
                            audio_captions_map[i] = caption_audio(tmp_aud_path, a_model, a_proc, audio_prompt)
                        except: audio_captions_map[i] = ""
                        finally: 
                            if os.path.exists(tmp_aud_path): os.unlink(tmp_aud_path)
                        prog_a.progress((i+1)/len(final_segments))
                    full_audio_clip.close(); del a_model, a_proc; clear_vram()
                else: st.error("Audio Load Error.")

            # === PHASE 3: VISION & MERGE ===
            status_box.info("👁️ **Phase 3/3: Vision Captioning & Merging**...")
            folder_name = project_name.strip() if project_name.strip() else f"dataset_{target_dur}s"
            out_dir = os.path.join(BASE_DIR, folder_name); os.makedirs(out_dir, exist_ok=True)
            
            # ➕ DODAJ TĘ LINIĘ PONIŻEJ:
            st.success(f"Found {len(final_segments)} clips. Saving to: {out_dir}") 
            # --------------------------

            v_model, v_proc = load_vision_model()
            video_f = VideoFileClip(video_path)
            prog_v = st.progress(0)

            for i, seg in enumerate(final_segments[:100]):
                base = f"clip_{i+1:03d}"; c_path = os.path.join(out_dir, f"{base}.mp4")
                cut_end = seg['end'] if not enable_hard_cut else (seg['start'] + target_dur)
                
                sub = video_f.subclipped(seg['start'], cut_end)
                if not keep_orig: sub = sub.resized(new_size=(out_width, out_height)).write_videofile(c_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                else: sub.write_videofile(c_path, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                
                vis_cap = caption_vision(c_path, "video", v_model, v_proc, lora_trigger, user_instruction, SELECTED_VISION_ID)
                
                if enable_hard_cut:
                    valid_words = [w['word'] for w in all_words_global if w['start'] >= seg['start'] and w['end'] <= cut_end]
                    speech = " ".join(valid_words).strip()
                    if not speech: speech = seg['text'].strip()
                else: speech = seg['text'].strip()
                
                aud_cap = audio_captions_map.get(i, "").strip()
                if aud_cap: final_text = f"{vis_cap} In the background, {aud_cap}. The character says: \"{speech}\""
                else: final_text = f"{vis_cap} The character says: \"{speech}\""
                
                final_text = final_text.replace("..", ".").replace("  ", " ").strip()
                
                with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f: f.write(final_text)
                with st.expander(f"✅ {base}"): st.video(c_path); st.info(f"**Caption:** {final_text}")
                prog_v.progress((i+1)/len(final_segments))

            video_f.close(); del v_model, v_proc; clear_vram()
            st.success("✅ DONE! v4.0 Pipeline Finished.")
            end_ts = time.time(); mins, secs = divmod(end_ts - start_ts, 60)
            timer_placeholder.success(f"⏱️ Total Time: {int(mins)}m {int(secs)}s")

        except Exception as e:
            st.error(f"Critical Error: {e}")
            import traceback; st.code(traceback.format_exc())
        finally:
            clear_vram(); 
            if os.path.exists(video_path): os.unlink(video_path)

# =================================================================================================
# MODE 2: BULK VIDEO CAPTIONER (v4.0 Fixed with Selection)
# =================================================================================================
elif app_mode == "📝 Bulk Video Captioner":
    if 'v_bulk_path' not in st.session_state: st.session_state['v_bulk_path'] = ""
    col_v, col_vbtn = st.columns([3, 1])
    with col_vbtn:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['v_bulk_path'] = sel; st.rerun()
    with col_v: v_dir = st.text_input("Folder Path:", value=st.session_state['v_bulk_path'])
    
    st.markdown("### 🛠️ Bulk Processing Options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_vis = st.checkbox("✅ Enable Vision Captioning (Qwen-VL)", value=True)
    with col_opt2:
        enable_speech = st.checkbox("✅ Enable Speech Transcription (WhisperX)", value=True)

    if st.button("🚀 START BULK CAPTIONING") and os.path.exists(v_dir):
        if not enable_vis and not enable_speech:
            st.error("Select at least one option!")
        else:
            start_ts = time.time()
            status_box = st.empty()
            videos = [f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".mkv"))]
            transcriptions = {}
            
            if not videos: st.warning("No videos found!")
            else:
                # 1. FAZA AUDIO (WHISPER)
                if enable_speech:
                    status_box.info(f"🎤 **Phase 1: Transcribing Audio for {len(videos)} clips...**")
                    try:
                        model_w = whisperx.load_model("large-v3", device, compute_type="float16", download_root=MODELS_DIR)
                        prog_a = st.progress(0)
                        for i, v_name in enumerate(videos):
                            full_p = os.path.join(v_dir, v_name)
                            audio = whisperx.load_audio(full_p)
                            result = model_w.transcribe(audio, batch_size=16)
                            full_text = " ".join([seg['text'].strip() for seg in result['segments']])
                            transcriptions[v_name] = full_text
                            prog_a.progress((i+1)/len(videos))
                        del model_w, audio; clear_vram()
                    except Exception as e: st.error(f"Whisper Error: {e}"); clear_vram()
                
                # 2. FAZA VISION & MERGE
                if enable_vis:
                    status_box.info(f"👁️ **Phase 2: Visual Captioning...**")
                    v_model, v_proc = load_vision_model() # Używamy nowej funkcji loadera z v4.0
                
                prog_main = st.progress(0)
                for i, v_name in enumerate(videos):
                    p = os.path.join(v_dir, v_name)
                    final_txt = ""
                    
                    # Vision Part
                    if enable_vis:
                        vis_cap = caption_vision(p, "video", v_model, v_proc, lora_trigger, user_instruction, SELECTED_VISION_ID)
                        final_txt += vis_cap
                    
                    # Speech Part
                    if enable_speech:
                        speech = transcriptions.get(v_name, "")
                        if speech:
                            separator = " The person says: " if enable_vis else "The person says: "
                            final_txt += f'{separator}"{speech}"'
                    
                    with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: 
                        f.write(final_txt.strip())
                    
                    prog_main.progress((i+1)/len(videos))
                
                if enable_vis:
                    del v_model, v_proc; clear_vram()
                
                status_box.empty()
                st.success("✅ DONE! Bulk Processing finished.")
                mins, secs = divmod(time.time() - start_ts, 60)
                st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")

# =================================================================================================
# MODE 3: IMAGE FOLDER CAPTIONER (Restored Standard Functionality)
# =================================================================================================
else: 
    if 'img_path' not in st.session_state: st.session_state['img_path'] = ""
    col_p, col_b = st.columns([3, 1])
    with col_b:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['img_path'] = sel; st.rerun()
    with col_p: img_dir = st.text_input("Path:", value=st.session_state['img_path'])
    
    if st.button("🚀 CAPTION FOLDER") and os.path.exists(img_dir):
        start_ts = time.time()
        v_model, v_proc = load_vision_model()
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".webp"))]
        
        if not imgs: st.warning("No images found!")
        else:
            prog = st.progress(0)
            for i, name in enumerate(imgs):
                p = os.path.join(img_dir, name)
                # Używamy tej samej funkcji caption_vision co do wideo, ale z typem 'image'
                cap = caption_vision(p, "image", v_model, v_proc, lora_trigger, user_instruction, SELECTED_VISION_ID)
                with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: f.write(cap)
                prog.progress((i+1)/len(imgs))
            
            del v_model, v_proc
            clear_vram()
            st.success("✅ DONE! Folder Captioning finished.")
            mins, secs = divmod(time.time() - start_ts, 60)
            st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")

st.markdown("---")
st.markdown("<center><b>Project maintained by Cyberbol</b></center>", unsafe_allow_html=True)