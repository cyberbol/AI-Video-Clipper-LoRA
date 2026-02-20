#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner - Unified Vision Engine
# Contributor: FNGarvin | License: MIT
# --------------------------------------------------------------------------------

import os
import gc
import torch
import base64
from threading import Thread
from .downloader import download_model
from PIL import Image
import numpy as np
import tempfile

# --- Lazy Imports for Heavy Libraries ---
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None

IMPORT_ERROR = None
try:
    try:
        # Transformers v5+
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        # Transformers v4.x
        from transformers import AutoModelForVision2Seq
        
    from transformers import AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer
    from qwen_vl_utils import process_vision_info
except Exception as e:
    AutoModelForVision2Seq = None
    IMPORT_ERROR = e

# --------------------------------------------------------------------------------

def sample_frames(video_path, num_frames=8):
    """Extracts evenly spaced frames from a video."""
    from moviepy.video.io.VideoFileClip import VideoFileClip
    clip = VideoFileClip(video_path)
    duration = clip.duration
    timestamps = np.linspace(0, duration, num_frames + 2)[1:-1] # Avoid start/end 
    frames = []
    for t in timestamps:
        frame_np = clip.get_frame(t)
        frames.append(Image.fromarray(frame_np))
    clip.close()
    return frames

# --------------------------------------------------------------------------------

class VisionEngine:
    """
    Unified entry point for Vision Models.
    Factory that returns the appropriate engine (GGUF or Transformers) based on config.
    """
    def __init__(self, model_config, device="cuda", models_dir="models", n_ctx=8192, n_gpu_layers=-1):
        self.config = model_config
        self.device = device
        self.models_dir = models_dir
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.engine = None
        
        backend = self.config.get("backend", "transformers")
        
        if backend == "gguf":
            self.engine = GGUFVisionEngine(
                repo_id=self.config["repo"],
                model_file=self.config["model"],
                projector_file=self.config["projector"],
                device=device,
                models_dir=models_dir,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers
            )
        else:
            self.engine = TransformersVisionEngine(
                model_id=self.config["id"],
                device=device,
                models_dir=models_dir
            )

    def load(self, log_callback=None):
        return self.engine.load(log_callback)
    
    def caption(self, *args, **kwargs):
        return self.engine.caption(*args, **kwargs)
        
    def clear(self):
        if self.engine:
            self.engine.clear()
        self.engine = None
        gc.collect()
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------

class GGUFVisionEngine:
    def __init__(self, repo_id, model_file, projector_file, device="cuda", models_dir="models", n_ctx=8192, n_gpu_layers=-1):
        self.repo_id = repo_id
        self.model_file = model_file
        self.projector_file = projector_file
        self.models_dir = models_dir
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self.chat_handler = None

    def load(self, log_callback=None):
        if Llama is None:
            raise ImportError("llama-cpp-python is missing. Install with CUDA support.")
            
        # Download/Verify Files
        targets = [self.model_file]
        if self.projector_file:
            targets.append(self.projector_file)
            
        model_root = download_model(self.repo_id, self.models_dir, specific_files=targets, log_callback=log_callback)
        
        m_path = os.path.normpath(os.path.join(model_root, self.model_file))
        
        if log_callback: log_callback(f"🧠 Loading GGUF: {self.model_file}...")
        
        # Verify paths exist
        if not os.path.exists(m_path): raise FileNotFoundError(f"Model file missing: {m_path}")
        
        p_path = None
        if self.projector_file:
            p_path = os.path.normpath(os.path.join(model_root, self.projector_file))
            if not os.path.exists(p_path): raise FileNotFoundError(f"Projector file missing: {p_path}")

        if p_path:
            try:
                self.chat_handler = Llava15ChatHandler(clip_model_path=p_path, verbose=False)
            except Exception as e:
                print(f"⚠️ Failed to init LlavaChatHandler: {e}")
                self.chat_handler = None
        else:
            self.chat_handler = None

        self.llm = Llama(
            model_path=m_path,
            chat_handler=self.chat_handler,
            n_ctx=self.n_ctx,  # Configurable
            n_gpu_layers=self.n_gpu_layers, # Configurable
            verbose=False    # Keep checking stdout clean
        )
        if log_callback: log_callback("✅ GGUF Engine Ready.")

    def caption(self, content_path, content_type, trigger, instruction, gen_config=None, stream_callback=None):
        if not self.llm: raise RuntimeError("Engine not loaded.")
        
        # Prompt Formatting
        if not instruction.strip(): instruction = "Describe this {type} in high detail."
        final_prompt = instruction.replace("{trigger}", trigger if trigger else "").replace("{type}", "video" if content_type == "video" else "image")
        
        # Content processing strategy
        is_video = (content_type == "video")
        force_frames = False
        
        if is_video:
            name_lower = self.model_file.lower()
            if "vl" not in name_lower or "gemma" in name_lower:
                force_frames = True
        
        imgs_b64 = []
        if is_video and force_frames:
            # Multi-frame sampling (Gemma style)
            import base64
            frames = sample_frames(content_path, num_frames=8)
            for pil_img in frames:
                # Convert PIL to base64
                from io import BytesIO
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                imgs_b64.append(b64)
            final_prompt = f"The following are 8 frames sampled from a video. {final_prompt}"
        else:
            # Single Image or One-Frame fallback
            imgs_b64.append(self._get_image_base64(content_path, content_type))

        # Build Messages
        content_list = []
        for b64 in imgs_b64:
             content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content_list.append({"type": "text", "text": final_prompt})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content_list}
        ]
        
        # Params
        params = {"max_tokens": 256, "temperature": 0.7, "top_p": 0.9}
        if gen_config: params.update({k: v for k, v in gen_config.items() if k in params})

        # Generate
        if stream_callback:
            response_iter = self.llm.create_chat_completion(messages=messages, stream=True, **params)
            full_text = ""
            for chunk in response_iter:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    full_text += delta["content"]
                    # Stream raw accumulated text; only post-process once at the end
                    stream_callback(full_text)
            return self._post_process(full_text, trigger)
        else:
            response = self.llm.create_chat_completion(messages=messages, **params)
            return self._post_process(response["choices"][0]["message"]["content"], trigger)

    def _get_image_base64(self, path, c_type):
        if c_type == "video":
            # Extract simple middle frame for now
            from moviepy.video.io.VideoFileClip import VideoFileClip
            try:
                clip = VideoFileClip(path)
                
                # Use a unique temp file to avoid collisions (Sourcery fix)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    frame_path = tmp.name
                
                clip.save_frame(frame_path, t=clip.duration / 2)
                clip.close()
                
                with open(frame_path, "rb") as f: 
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
                return b64
            except Exception as e:
                print(f"Frame extraction failed: {e}")
                import traceback
                traceback.print_exc()
                return ""
        else:
            with open(path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")

    def _post_process(self, text, trigger):
        text = text.replace("**", "").replace("##", "").strip()
        if trigger and trigger.lower() not in text.lower():
            text = f"{trigger}, {text}"
        return text

    def clear(self):
        self.llm = None
        self.chat_handler = None
        gc.collect()
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------

class TransformersVisionEngine:
    def __init__(self, model_id, device="cuda", models_dir="models"):
        self.model_id = model_id
        self.device = device
        self.models_dir = models_dir
        self.model = None
        self.processor = None

    def load(self, log_callback=None):
        if AutoModelForVision2Seq is None:
            if IMPORT_ERROR:
                raise ImportError(f"Transformers import failed: {IMPORT_ERROR}")
            raise ImportError("Transformers not installed.")
            
        # Force Online for Download (using robust downloader)
        path = download_model(self.model_id, self.models_dir, log_callback=log_callback)
        
        if log_callback: log_callback(f"🧠 Loading Transformers Model: {self.model_id}...")
        
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                path,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            if log_callback: log_callback("✅ Transformers Engine Ready.")
        except Exception as e:
            if log_callback: log_callback(f"❌ Load Failed: {e}")
            raise e

    def caption(self, content_path, content_type, trigger, instruction, gen_config=None, stream_callback=None):
        if not self.model: raise RuntimeError("Engine not loaded.")
        
        # Prompt
        if not instruction.strip(): instruction = "Describe this {type} in detail."
        final_prompt = instruction.replace("{trigger}", trigger if trigger else "").replace("{type}", "video" if content_type == "video" else "image")

        # Gemma / Generic Video Handling
        is_gemma = "gemma" in self.model_id.lower()
        
        if is_gemma and content_type == "video":
            # Gemma 2/3 video path -> 8 sampled frames
            frames = sample_frames(content_path, num_frames=8)
            content_list = []
            for frame in frames:
                content_list.append({"type": "image", "image": frame})
            
            final_prompt = f"The following are 8 frames sampled from a video. {final_prompt}"
            content_list.append({"type": "text", "text": final_prompt})
            messages = [{"role": "user", "content": content_list}]
        else:
            # Standard Qwen / Native
            messages = [{"role": "user", "content": [
                {"type": content_type, content_type: content_path, "max_pixels": 360*420, "fps": 1.0},
                {"type": "text", "text": final_prompt}
            ]}]
        
        # Preprocessing
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generation Params
        params = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        if gen_config: 
            # Transformers uses max_new_tokens, config might use max_tokens
            if "max_tokens" in gen_config: params["max_new_tokens"] = gen_config["max_tokens"]
            params.update({k:v for k,v in gen_config.items() if k in params})

        if stream_callback:
            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            params["streamer"] = streamer
            
            thread = Thread(target=self.model.generate, kwargs={**inputs, **params})
            thread.start()
            
            full_text = ""
            for new_text in streamer:
                full_text += new_text
                stream_callback(self._post_process(full_text, trigger))
            
            thread.join()
            return self._post_process(full_text, trigger)
        else:
            generated_ids = self.model.generate(**inputs, **params)
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            return self._post_process(output_text, trigger)

    def _post_process(self, text, trigger):
        text = text.replace("**", "").replace("##", "").strip()
        if trigger and trigger.lower() not in text.lower():
            text = f"{trigger}, {text}"
        return text

    def clear(self):
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()

# Utilities
def scan_local_gguf_models(models_dir):
    """Scans for GGUF pairs (model + projector) in models_dir."""
    discovered = {}
    if not os.path.exists(models_dir): return discovered
    
    for root, dirs, files in os.walk(models_dir):
        ggufs = [f for f in files if f.endswith(".gguf") and not f.lower().startswith("mmproj-")]
        projectors = [f for f in files if f.lower().startswith("mmproj-") and f.endswith(".gguf")]
        
        if ggufs and projectors:
            rel_root = os.path.relpath(root, models_dir).replace("\\", "/")
            # Use the first projector found (Shared strategy)
            proj_file = projectors[0]
            
            for g in ggufs:
                # Clean Label: Filename without extension
                clean_name = g.replace(".gguf", "")
                label = f"{clean_name} (local)"
                
                discovered[label] = {
                    "backend": "gguf",
                    "repo": rel_root, 
                    "model": g,
                    "projector": proj_file
                }
    return discovered

# EOF vision_engine.py
