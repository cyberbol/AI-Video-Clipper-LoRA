# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner (v5.0 Staging)
# 🏆 CREDITS: Cyberbol (Logic), FNGarvin (Engine), WildSpeaker (5090 Fix)
# --------------------------------------------------------------------------------

import os
import sys

# --- 1. BOOTSTRAP & PATCHES ---
# 🚨 CRITICAL: Patches must apply BEFORE heavy imports (Torch, WhisperX)
try:
    from modules import patches
    patches.apply_patches()
except ImportError:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Basic fallbacks if patches.py is missing
    pass

import streamlit as st
import whisperx
from moviepy import VideoFileClip, AudioFileClip
import tempfile
import torch
import gc
import time
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
import logging
import shutil

# Suppress Streamlit's "missing ScriptRunContext" warning in threads
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

# 5090 Fix (In-line backup if patches failed or specifically for this file)
if not hasattr(torch, "_patched_for_5090"):
    _orig_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _safe_load
    torch._patched_for_5090 = True

# --- EXPORT MODULES TO PATH ---
# Ensure modules can be imported if running from a subdir (future proofing)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.vision_engine import VisionEngine, scan_local_gguf_models
from modules.audio_engine import AudioEngine
from modules.downloader import download_model

# --- 2. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# --- 3. UI CONFIG ---
st.set_page_config(page_title="AI Clipper v5.0", layout="wide")
st.title("👁️🐧👂 AI Video Clipper & LoRA Captioner")
st.markdown("v5.0 | **[Cyberbol](https://github.com/cyberbol/)** (Project Creator) | **[FNG](https://github.com/FNGarvin/)** (Engine) | **WildSpeaker** (Has a 5090!)")

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
    "Transformer: Qwen2-VL-7B (High Quality)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-7B-Instruct"
    },
    "Transformer: Qwen2-VL-2B (Fast)": {
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

model_label = st.sidebar.selectbox("Vision Model:", list(model_options.keys()), index=2)
SELECTED_MODEL = model_options[model_label]
SELECTED_VISION_ID = SELECTED_MODEL.get("id", SELECTED_MODEL.get("model")) # Fallback ID

# Audio Model ID (Fixed for now, can be modularized later)
SELECTED_AUDIO_ID = "mlinmg/Qwen-2-Audio-Instruct-dynamic-fp8"

st.sidebar.divider()
st.sidebar.markdown("### 📝 Instructions")

# PROMPTY
default_prompt = "Describe this {type} in detail for a dataset. Main subject: {trigger}. Describe the action, camera movement, lighting, atmosphere, and background."
user_instruction = st.sidebar.text_area("Vision Prompt:", value=default_prompt, height=150)
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

audio_prompt_default = "Describe the sound atmosphere and music mood in one short, non-technical sentence. Do not mention BPM or keys."
audio_prompt = st.sidebar.text_area("Audio Prompt (Qwen2-Audio):", value=audio_prompt_default, height=60)

# --- SIDEBAR: ADVANCED OPTIONS ---
with st.sidebar.expander("🛠️ Advanced Generation Options"):
    gen_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, help="Controls randomness. Higher = creative/chaotic. Lower = focused/deterministic.")
    gen_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05, help="Nucleus sampling. Restricts generation to the top P probability mass. Lower = more focused.")
    gen_max_tokens = st.number_input("Max New Tokens", 64, 2048, 256, help="Maximum number of tokens to generate. Increase for longer descriptions.")
    
    st.markdown("#### ⚙️ Model Loading (GGUF)")
    n_ctx_val = st.number_input("Context Window (n_ctx)", 2048, 32768, 8192, step=1024, help="Lower if VRAM is tight.")
    n_gpu_layers_val = st.number_input("GPU Layers (n_gpu_layers)", -1, 100, -1, help="-1 = All layers to GPU. Reduce if OOM.")
    
GEN_CONFIG = {
    "temperature": gen_temp,
    "top_p": gen_top_p,
    "max_tokens": gen_max_tokens
}

# --- 4. HELPERS ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- 5. MODEL LOADERS (MODULARIZED) ---
def load_vision_engine():
    """Unified loading using VisionEngine class"""
    # Check if model changed or loading config changed
    current_config = f"{str(SELECTED_MODEL)}_{n_ctx_val}_{n_gpu_layers_val}"
    
    if 'vision_engine' not in st.session_state or \
       st.session_state.get('last_model_config') != current_config or \
       (st.session_state['vision_engine'] and st.session_state['vision_engine'].engine is None):
        
        # Clear old
        if 'vision_engine' in st.session_state and st.session_state['vision_engine']:
            st.session_state['vision_engine'].clear()
            
        with st.status(f"🚀 Initializing Vision Engine ({model_label})...", expanded=True) as status:
            engine = VisionEngine(SELECTED_MODEL, device=device, models_dir=MODELS_DIR, n_ctx=n_ctx_val, n_gpu_layers=n_gpu_layers_val)
            engine.load(log_callback=status.write)
            status.update(label="✅ Vision Engine Ready!", state="complete", expanded=False)
            
        st.session_state['vision_engine'] = engine
        st.session_state['last_model_config'] = current_config
    
    return st.session_state['vision_engine']

def load_audio_engine():
    """Unified loading using AudioEngine class"""
    if 'audio_engine' not in st.session_state:
        st.session_state['audio_engine'] = None

    # Lazy load or check existence? 
    # For now, let's just return the instance, create if needed
    if not st.session_state['audio_engine']:
        engine = AudioEngine(SELECTED_AUDIO_ID, device=device, models_dir=MODELS_DIR)
        # We don't auto-load here to save VRAM, load on demand in the loop
        st.session_state['audio_engine'] = engine
        
    return st.session_state['audio_engine']

def select_folder_dialog():
    root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root); root.destroy()
    return folder_path

# =================================================================================================
# MODE 1: VIDEO AUTO-CLIPPER (v5.0 Merged Logic)
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
        
        video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read()); video_path = tmp.name
            

            # === PHASE 0: CLEAN SLATE ===
            # Force unload any existing engines to free VRAM for Whisper
            if 'vision_engine' in st.session_state and st.session_state['vision_engine']:
                st.session_state['vision_engine'].clear()
            if 'audio_engine' in st.session_state and st.session_state['audio_engine']:
                st.session_state['audio_engine'].clear()
            clear_vram()

            # === PHASE 1: WHISPER X ===
            print(f"\n🚀 [Phase 1] Speech Analysis (WhisperX) started on {video_path}...")
            status_box.info("🚀 **Phase 1/3: Speech Analysis (WhisperX)**...")
            
             # Use downloader to ensure model is present and get local path
            wx_repo = "Systran/faster-whisper-large-v3"
            wx_path = os.path.normpath(os.path.join(MODELS_DIR, wx_repo))
            
            # Robust Check
            essential_wx = ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]
            if not all(os.path.exists(os.path.join(wx_path, f)) for f in essential_wx):
                with st.spinner("Ensuring Whisper Model (Downloader Active)..."):
                    wx_path = download_model(wx_repo, MODELS_DIR, specific_files=essential_wx, log_callback=status_box.text)

            check_clip = VideoFileClip(video_path); video_duration = check_clip.duration; check_clip.close(); del check_clip

            model_w = whisperx.load_model(wx_path, device, compute_type="float16")
            audio_source = whisperx.load_audio(video_path)
            result = model_w.transcribe(audio_source, batch_size=16)
            
            # Alignment (Optional)
            try:
                align_model_dir = os.path.join(MODELS_DIR, "PyTorch")
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir=align_model_dir)
                result = whisperx.align(result["segments"], model_a, metadata, audio_source, device)
                del model_a, metadata
            except Exception as e:
                logging.warning(f"Alignment Failed: {e}") 
                status_box.warning(f"⚠️ Additional Language models not available (Language: {result['language']}) - using standard timestamps.")
            
            all_words_global = []
            for s in result["segments"]:
                if 'words' in s: all_words_global.extend(s['words'])

            del model_w, audio_source
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
                print(f"\n👂 [Phase 2] Audio Analysis started ({len(final_segments)} clips)...")
                status_box.info(f"👂 **Phase 2/3: Audio Analysis ({len(final_segments)} clips)**...")
                
                # Auto-Download Qwen2-Audio if missing
                # It's better to ensure files locally for offline support
                with st.spinner("Ensuring Audio Model (Downloader Active)..."):
                    download_model(SELECTED_AUDIO_ID, MODELS_DIR, log_callback=status_box.text)

                a_engine = load_audio_engine()
                # Ensure loaded
                if not a_engine.model:
                    with st.spinner("Loading Audio Engine..."):
                        a_engine.load(log_callback=status_box.text)
                
                if a_engine.model:
                    try:
                        prog_a = st.progress(0)
                        full_audio_clip = AudioFileClip(video_path)
                        for i, seg in enumerate(final_segments[:100]):
                            cut_end = seg['end'] if not enable_hard_cut else (seg['start'] + target_dur)
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_aud:
                                sub_a = full_audio_clip.subclipped(seg['start'], cut_end)
                                sub_a.write_audiofile(tmp_aud.name, logger=None)
                                tmp_aud_path = tmp_aud.name
                            try:
                                audio_captions_map[i] = a_engine.caption_audio(tmp_aud_path, audio_prompt)
                            except Exception as e: 
                                print(f"Audio Cap Error: {e}")
                                audio_captions_map[i] = ""
                            finally: 
                                if os.path.exists(tmp_aud_path): os.unlink(tmp_aud_path)
                            prog_a.progress((i+1)/len(final_segments))
                        full_audio_clip.close(); 
                        # FREE VRAM: Audio Engine is done, clear it for Vision
                        a_engine.clear() 
                        clear_vram()
                    except Exception as e:
                        # Catch-all cleanup to prevent OOM on crash
                        if 'a_engine' in locals(): a_engine.clear()
                        clear_vram()
                        raise e
                else: st.error("Audio Engine Load Error.")

            # === PHASE 3: VISION & MERGE ===
            print(f"\n👁️ [Phase 3] Vision Captioning & Merging started...")
            status_box.info("👁️ **Phase 3/3: Vision Captioning & Merging**...")
            folder_name = project_name.strip() or f"dataset_{target_dur}s"
            out_dir = os.path.join(BASE_DIR, folder_name); os.makedirs(out_dir, exist_ok=True)
            
            st.success(f"Found {len(final_segments)} clips. Saving to: {out_dir}") 
            
            v_engine = load_vision_engine()
            video_f = VideoFileClip(video_path)
            prog_v = st.progress(0)

            for i, seg in enumerate(final_segments[:100]):
                base = f"clip_{i+1:03d}"; c_path = os.path.join(out_dir, f"{base}.mp4")
                cut_end = seg['end'] if not enable_hard_cut else (seg['start'] + target_dur)
                
                sub = video_f.subclipped(seg['start'], cut_end)
                if not keep_orig: sub = sub.resized(new_size=(out_width, out_height)).write_videofile(c_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                else: sub.write_videofile(c_path, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                
                # Streaming UI
                stream_box = st.empty()
                def on_token(text):
                    stream_box.markdown(f"**📝 Generating:** {text}")

                vis_cap = v_engine.caption(c_path, "video", lora_trigger, user_instruction, 
                                           gen_config=GEN_CONFIG, stream_callback=on_token)
                stream_box.empty()
                
                if enable_hard_cut:
                    valid_words = [w['word'] for w in all_words_global if w['start'] >= seg['start'] and w['end'] <= cut_end]
                    speech = " ".join(valid_words).strip()
                    speech = " ".join(valid_words).strip() or seg['text'].strip()
                else: speech = seg['text'].strip()
                
                if aud_cap := audio_captions_map.get(i, "").strip():
                    final_text = f"{vis_cap} In the background, {aud_cap}. The character says: \"{speech}\""
                else:
                    final_text = f"{vis_cap} The character says: \"{speech}\""
                
                final_text = final_text.replace("..", ".").replace("  ", " ").strip()
                
                with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f: f.write(final_text)
                with st.expander(f"✅ {base}"): st.video(c_path); st.info(f"**Caption:** {final_text}")
                prog_v.progress((i+1)/len(final_segments))

            video_f.close(); 
            clear_vram()
            st.success("✅ DONE! v5.0 Pipeline Finished.")
            end_ts = time.time(); mins, secs = divmod(end_ts - start_ts, 60)
            timer_placeholder.success(f"⏱️ Total Time: {int(mins)}m {int(secs)}s")

        except Exception as e:
            st.error(f"Critical Error: {e}")
            import traceback; st.code(traceback.format_exc())
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)

# =================================================================================================
# MODE 2: BULK VIDEO CAPTIONER
# =================================================================================================
elif app_mode == "📝 Bulk Video Captioner":
    if HAS_TKINTER:
        # --- DESKTOP MODE (Folder Selection) ---
        if 'v_bulk_path' not in st.session_state: st.session_state['v_bulk_path'] = ""
        col_v, col_vbtn = st.columns([3, 1])
        with col_vbtn:
            if st.button("📂 Select Folder"):
                sel = select_folder_dialog()
                if sel: st.session_state['v_bulk_path'] = sel; st.rerun()
        with col_v: v_dir = st.text_input("Folder Path:", value=st.session_state['v_bulk_path'])
        uploaded_files = None
    else:
        # --- HEADLESS/DOCKER MODE (File Upload) ---
        st.info("☁️ **Cloud Mode Detected**: Upload videos directly below.")
        v_dir = None
        uploaded_files = st.file_uploader("Upload Videos for Batch Processing", type=['mp4', 'mkv', 'mov'], accept_multiple_files=True)

    st.markdown("### 🛠️ Bulk Processing Options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_vis = st.checkbox("✅ Enable Vision Captioning (Qwen-VL/GGUF)", value=True)
    with col_opt2:
        enable_speech = st.checkbox("✅ Enable Speech Transcription (WhisperX)", value=True)

    # Trigger Logic
    start_processing = False
    if HAS_TKINTER:
        if st.button("🚀 START BULK CAPTIONING") and v_dir and os.path.exists(v_dir):
            start_processing = True
    else:
        if uploaded_files and st.button("🚀 PROCESS UPLOADED VIDEOS"):
            start_processing = True

    if start_processing:
        # Setup Temp Dir for Uploads (if applicable)
        temp_upload_dir = None
        if not HAS_TKINTER and uploaded_files:
            temp_upload_dir = tempfile.mkdtemp(prefix="bulk_upload_")
            v_dir = temp_upload_dir
            st.info(f"📦 Staging {len(uploaded_files)} files...")
            for uf in uploaded_files:
                with open(os.path.join(temp_upload_dir, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
        
        try:
            if not enable_vis and not enable_speech:
                st.error("Select at least one option!")
            else:
                start_ts = time.time()
                status_box = st.empty()
                videos = [f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".mkv", ".mov"))]
                transcriptions = {}
                
                if not videos: st.warning("No videos found!")
                else:
                    # 1. FAZA AUDIO (WHISPER)
                    if enable_speech:
                        status_box.info(f"🎤 **Phase 1: Transcribing Audio for {len(videos)} clips...**")
                        try:
                            # Ensure model
                            wx_repo = "Systran/faster-whisper-large-v3"

                            wx_path = os.path.normpath(os.path.join(MODELS_DIR, wx_repo))
                            essential_wx = ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]
                            if not all(os.path.exists(os.path.join(wx_path, f)) for f in essential_wx):
                                 with st.spinner("Ensuring Whisper Model..."):
                                     wx_path = download_model(wx_repo, MODELS_DIR, specific_files=essential_wx)

                            model_w = whisperx.load_model(wx_path, device, compute_type="float16")
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
                    v_engine = load_vision_engine()
                
                prog_main = st.progress(0)
                for i, v_name in enumerate(videos):
                    p = os.path.join(v_dir, v_name)
                    final_txt = ""
                    
                    # Vision Part
                    if enable_vis:
                        stream_box = st.empty()
                        def on_token(text):
                            stream_box.markdown(f"**Processing {v_name}:** {text}")

                        vis_cap = v_engine.caption(p, "video", lora_trigger, user_instruction, 
                                                   gen_config=GEN_CONFIG, stream_callback=on_token)
                        stream_box.empty()
                        final_txt += vis_cap
                    
                    # Speech Part
                    if enable_speech:
                        speech = transcriptions.get(v_name, "")
                        if speech:
                            separator = " The person says: " if enable_vis else "The person says: "
                            final_txt += f'{separator}"{speech}"'
                    
                    with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: 
                        f.write(final_txt.strip())
                        
                    # IF HEADLESS: Provide Download Button for Result TXT immediately (since files are ephemeral)
                    if not HAS_TKINTER:
                        with open(os.path.splitext(p)[0] + ".txt", "r", encoding="utf-8") as f:
                            st.download_button(f"⬇️ Download Caption ({v_name})", f.read(), file_name=f"{v_name}.txt")
                    
                    prog_main.progress((i+1)/len(videos))

                st.success("✅ Bulk Processing Complete!")
                
                # IF HEADLESS: Zip Everything for easy download
                if not HAS_TKINTER and temp_upload_dir:
                     import shutil
                     zip_path = shutil.make_archive(os.path.join(tempfile.gettempdir(), "captions_batch"), 'zip', temp_upload_dir)
                     with open(zip_path, "rb") as f:
                         st.download_button("📦 DOWNLOAD ALL CAPTIONS (ZIP)", f, file_name="captions_batch.zip")

        finally:
            # Cleanup Temp Uploads
            if temp_upload_dir and os.path.exists(temp_upload_dir):
                import shutil
                shutil.rmtree(temp_upload_dir)
                
                if enable_vis:
                    # Should we clear? Not strictly necessary if we want to keep it warm, but safer for VRAM
                    # v_engine.clear() 
                    clear_vram()
                
                status_box.empty()
                st.success("✅ DONE! Bulk Processing finished.")
                mins, secs = divmod(time.time() - start_ts, 60)
                st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")

# =================================================================================================
# MODE 3: IMAGE FOLDER CAPTIONER
# =================================================================================================
else: 
    if HAS_TKINTER:
        if 'img_path' not in st.session_state: st.session_state['img_path'] = ""
        col_p, col_b = st.columns([3, 1])
        with col_b:
            if st.button("📂 Select Folder"):
                if sel := select_folder_dialog():
                    st.session_state['img_path'] = sel
                    st.rerun()
        with col_p: img_dir = st.text_input("Path:", value=st.session_state['img_path'])
        uploaded_images = None
    else:
        st.info("☁️ **Cloud Mode Detected**: Upload images directly below.")
        img_dir = None
        uploaded_images = st.file_uploader("Upload Images for Batch Captioning", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    # Trigger Logic
    start_img_processing = False
    if HAS_TKINTER:
        if st.button("🚀 CAPTION FOLDER") and img_dir and os.path.exists(img_dir):
            start_img_processing = True
    else:
        if uploaded_images and st.button("🚀 PROCESS UPLOADED IMAGES"):
            start_img_processing = True

    if start_img_processing:
        start_ts = time.time()
        temp_img_dir = None
        
        try:
            if not HAS_TKINTER and uploaded_images:
                temp_img_dir = tempfile.mkdtemp()
                img_dir = temp_img_dir
                for uploaded_file in uploaded_images:
                    with open(os.path.join(temp_img_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())

            v_engine = load_vision_engine()
            imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            
            if not imgs:
                st.warning("No images found!")
                st.stop()
            
            prog = st.progress(0)
            for i, name in enumerate(imgs):
                p = os.path.join(img_dir, name)
                
                stream_box = st.empty()
                def on_token(text):
                    stream_box.markdown(f"**{name}:** {text}")
                    
                cap = v_engine.caption(p, "image", lora_trigger, user_instruction, 
                                       gen_config=GEN_CONFIG, stream_callback=on_token)
                stream_box.empty()
                
                txt_path = os.path.splitext(p)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f: 
                    f.write(cap)
                
                # IF HEADLESS: Provide Download Button
                if not HAS_TKINTER:
                    st.image(p, width=200)
                    st.download_button(f"⬇️ Download Caption ({name})", cap, file_name=os.path.basename(txt_path))
                
                prog.progress((i+1)/len(imgs))
                
            st.success("✅ DONE! Folder finished.")
            
            # IF HEADLESS: Zip Everything
            if not HAS_TKINTER and temp_img_dir:
                 zip_path = shutil.make_archive(os.path.join(tempfile.gettempdir(), "image_captions_batch"), 'zip', temp_img_dir)
                 with open(zip_path, "rb") as f:
                     st.download_button("📦 DOWNLOAD ALL IMAGE CAPTIONS (ZIP)", f, file_name="image_captions_batch.zip")
            
            mins, secs = divmod(time.time() - start_ts, 60)
            st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")

        finally:
            if temp_img_dir and os.path.exists(temp_img_dir):
                shutil.rmtree(temp_img_dir)

st.markdown("---")
st.markdown("<div style='text-align: center'><a href='https://github.com/cyberbol/AI-Video-Clipper-LoRA'>An Open Source Project</a></div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<div style='text-align: center'><a href='https://github.com/cyberbol/AI-Video-Clipper-LoRA'>An Open Source Project</a></div>", unsafe_allow_html=True)
