import torch
import librosa
import numpy as np
import logging
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

class AudioEngine:
    def __init__(self, model_id, device="cuda", models_dir="models"):
        self.model_id = model_id
        self.device = device
        self.models_dir = models_dir
        self.model = None
        self.processor = None

    def load(self, log_callback=None):
        if log_callback:
            log_callback(f"⏳ Loading Audio Intelligence Engine ({self.model_id})...")
        try:
            # Check for local path first to force offline mode if available
            import os
            local_path = os.path.join(self.models_dir, self.model_id)
            load_path = local_path if os.path.exists(local_path) else self.model_id
            
            if load_path == local_path:
                if log_callback: log_callback(f"📂 Loading from local cache: {local_path}")

            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                load_path, 
                torch_dtype="auto", 
                device_map="auto", 
                attn_implementation="sdpa", 
                low_cpu_mem_usage=True,
                cache_dir=self.models_dir
            )
            self.processor = AutoProcessor.from_pretrained(load_path, cache_dir=self.models_dir)
            if log_callback:
                log_callback("✅ Audio Engine Ready!")
            return True
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Failed to load Qwen2-Audio: {e}")
            logging.error(f"Failed to load Qwen2-Audio: {e}")
            return False

    def caption_audio(self, audio_path, custom_prompt):
        if not self.model or not self.processor:
            raise RuntimeError("Audio Engine not loaded.")

        y, sr = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        conversation = [
            {'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {'type': 'text', 'text': custom_prompt}]}
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=y, sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=48)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"[Audio] Generated: '{response}'")

        response = response.replace("\n", " ").strip()
        if response.endswith("."): 
            response = response[:-1]
        return response

    def clear(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
