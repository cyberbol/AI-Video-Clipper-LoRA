import os
import sys

# --- EMERGENCY BOOTSTRAP: CACHE & PATCHES ---
# This MUST happen before any AI libraries are loaded.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OTEL_SDK_DISABLED"] = "true"

try:
    import patches
    patches.apply_patches()
except (ImportError, AttributeError):
    pass
# ---------------------------------------------

import streamlit as st
import whisperx
from moviepy import VideoFileClip
import tempfile
import torch
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

# --- WINDOWS FIX (STILL HERE FOR CONTEXT) ---
# Zapobiega błędom biblioteki OMP na Windowsie
# (Already set above, but kept in place to preserve original file structure)

# --- App Configuration ---
st.set_page_config(page_title="Universal LoRA Dataset Creator", layout="wide")
st.title("👁️🐧 AI Video Clipper & LoRA Captioner")

# --- Hardware Check ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ AI Engine Settings")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"GPU: **{gpu_name}**")
else:
    st.sidebar.error("CUDA not detected!")

st.sidebar.divider()

# --- MODEL SELECTION ---
st.sidebar.subheader("🧠 Vision Model Selector")
model_choice = st.sidebar.radio(
    "Choose Model Version:",
    ["Qwen2-VL-7B (Best Quality)", "Qwen2-VL-2B (Fastest)"],
    index=0,
    help="7B is smarter but slower (15GB). 2B is very fast but less detailed (5GB)."
)

if "2B" in model_choice:
    SELECTED_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    st.sidebar.info("🚀 **Mode: Speed**\nGood for simple actions/clothing.")
else:
    SELECTED_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    st.sidebar.info("💎 **Mode: Quality**\nBest for details and complex scenes.")

st.sidebar.divider()
st.sidebar.markdown(f"📂 **Models Storage:**\n`{MODELS_DIR}`")

# --- Memory Management ---
def clear_vram():
    """Force clean GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

# --- Processing Functions ---

def run_whisper_processing(video_path, target_len, tol_min, tol_max):
    """Phase 1: Audio Only."""
    st.info("1/4 ⏳ Loading Whisper Large-v3...")
    
    compute_type = "float16"
    # WhisperX ładujemy z download_root ustawionym na nasz folder models
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root=MODELS_DIR)
    
    st.info("1/4 ⏳ Audio analysis...")
    audio = whisperx.load_audio(video_path)
    # Batch size 8 is safe for Windows
    result = model.transcribe(audio, batch_size=8)
    
    # CLEANUP WHISPER - zwalniamy VRAM po analizie audio
    del model
    clear_vram()
    st.toast("Whisper unloaded from VRAM.")
    
    st.info(f"2/4 ⏳ Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    del model_a
    clear_vram()
    
    found_clips = []
    min_dur = float(target_len) - float(tol_min)
    max_dur = float(target_len) + float(tol_max)
    if min_dur < 0: min_dur = 0.1
    
    for seg in result["segments"]:
        start = seg['start']
        end = seg['end']
        duration = end - start
        if min_dur <= duration <= max_dur:
            found_clips.append((start, end, seg['text']))
            
    return found_clips

def load_vision_model_now(model_id):
    """Phase 2: Load Vision Model (Dynamic Selection)."""
    st.info(f"3/4 ⏳ Loading Vision Model: {model_id}...")
    
    # Qwen użyje HF_HOME zdefiniowanego na górze skryptu
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_vision_caption(video_path, model, processor, trigger_word):
    # Budowanie instrukcji (Prompt)
    if trigger_word and trigger_word.strip():
        system_instruction = (
            f"TASK: Describe this video for a LoRA training dataset.\n"
            f"IMPORTANT: The main subject is UNIQUELY IDENTIFIED as '{trigger_word}'.\n"
            f"RULES:\n"
            f"1. You MUST refer to the subject as '{trigger_word}' immediately.\n"
            f"2. Do NOT use generic terms like 'woman', 'man', 'girl', 'boy', 'person' for the main subject.\n"
            f"3. Example start: 'A medium shot of {trigger_word} wearing...'\n"
            f"4. Describe clothing, background, and actions in detail.\n"
            f"5. Do not describe the audio."
        )
    else:
        system_instruction = (
            "TASK: Describe this video for a LoRA training dataset.\n"
            "RULES:\n"
            "1. Start with the camera angle (e.g., 'A close-up of...').\n"
            "2. Describe the subject naturally (e.g., 'a young woman', 'a man').\n"
            "3. Describe clothing, background, and actions in detail.\n"
            "4. Do not describe the audio."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420, 
                    "fps": 1.0, 
                },
                {"type": "text", "text": system_instruction},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=384)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    final_text = output_text[0]

    # --- SAFETY NET (BEZPIECZNIK DLA MODELU 2B) ---
    # Modele 2B czasem ignorują instrukcje. Tutaj sprawdzamy:
    # Czy użytkownik podał trigger word? Oraz czy NIE ma go w tekście wynikowym?
    # Jeśli tak - dopisujemy go siłą na początku.
    if trigger_word and trigger_word.strip():
        if trigger_word not in final_text:
            final_text = f"{trigger_word}, {final_text}"
            
    return final_text

# --- UI ---
uploaded_file = st.file_uploader("Upload Video File (MP4, MKV)", type=["mp4", "mkv", "mov"])
st.divider()

st.subheader("Dataset Settings")
lora_trigger = st.text_input("Trigger Word (Optional)", placeholder="e.g., prstxxx")
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1: target_sec = st.number_input("Target Duration (s)", 2.0, 60.0, 5.0, 0.5)
with col2: out_width = st.number_input("Width (px)", 100, 3840, 1024)
with col3: out_height = st.number_input("Height (px)", 100, 2160, 1024)
with col4: out_fps = st.number_input("FPS", 10, 60, 24)

t_col1, t_col2 = st.columns(2)
with t_col1: tol_minus = st.number_input("Tolerance Minus (-)", 0.0, 5.0, 0.5, 0.1)
with t_col2: tol_plus = st.number_input("Tolerance Plus (+)", 0.0, 10.0, 1.0, 0.1)

final_min = target_sec - tol_minus
final_max = target_sec + tol_plus
st.info(f"ℹ️ Searching clips between **{final_min:.2f}s** and **{final_max:.2f}s**")

if uploaded_file and st.button("START PROCESS 🚀"):
    # Fix for file handle on Windows (zapisujemy uploadowany plik tymczasowo)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    
    try:
        # Phase 1: Audio Processing (Whisper)
        clips = run_whisper_processing(video_path, target_sec, tol_minus, tol_plus)
        
        if not clips:
            st.warning("No clips found within the specified duration criteria.")
        else:
            trigger_info = f"Trigger: '{lora_trigger}'" if lora_trigger else "Trigger: NONE"
            st.success(f"Found {len(clips)} clips! {trigger_info}. Processing vision using: {model_choice}...")
            
            # Phase 2: Vision Processing (Qwen)
            vision_model, vision_processor = load_vision_model_now(SELECTED_MODEL_ID)
            
            output_dir = "dataset"
            os.makedirs(output_dir, exist_ok=True)
            
            video = VideoFileClip(video_path)
            progress_bar = st.progress(0)
            
            limit_clips = 100 # Limit dla bezpieczeństwa, można zwiększyć
            for i, (start, end, audio_text) in enumerate(clips[:limit_clips]):
                base_filename = f"{i+1:03d}"
                clip_filename = f"{base_filename}.mp4"
                out_vid_path = os.path.join(output_dir, clip_filename)
                
                # Cięcie wideo (MoviePy)
                new_clip = video.subclipped(start, end)
                new_clip = new_clip.resized(new_size=(out_width, out_height))
                new_clip.write_videofile(out_vid_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                
                # Generowanie opisu (Vision)
                visual_description = generate_vision_caption(out_vid_path, vision_model, vision_processor, lora_trigger)
                final_caption = f"{visual_description} The person says: \"{audio_text.strip()}\""
                
                # Zapis pliku tekstowego
                txt_filename = f"{base_filename}.txt"
                out_txt_path = os.path.join(output_dir, txt_filename)
                with open(out_txt_path, "w", encoding="utf-8") as f:
                    f.write(final_caption)
                
                # Wyświetlanie w interfejsie
                with st.expander(f"Clip {base_filename}", expanded=True):
                    col_vid, col_txt = st.columns([1, 1])
                    with col_vid:
                        # Używamy ścieżki absolutnej dla Streamlit Video playera
                        abs_vid_path = os.path.abspath(out_vid_path)
                        st.video(abs_vid_path)
                    with col_txt:
                        st.text_area("Caption", final_caption, height=200)
                
                progress_bar.progress((i + 1) / min(len(clips), limit_clips))
            
            video.close()
            # Sprzątanie po modelu Vision
            del vision_model
            del vision_processor
            clear_vram()
            st.success(f"Done! Check folder: {output_dir}")
            
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        
    finally:
        # Usuwamy plik tymczasowy wideo
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass

# --- Cleanup Button (Sidebar) ---
if st.sidebar.button("🧹 Clear output folder"):
    output_dir = "dataset"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        st.sidebar.success("Folder cleaned!")

st.markdown("---")
st.markdown("<center>Created by: Cyberbol</center>", unsafe_allow_html=True)