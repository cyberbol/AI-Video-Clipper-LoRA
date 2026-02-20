# AI Video Clipper & LoRA Captioner - Compatibility Patches

import torch
import torchaudio
import warnings
import os

def apply_patches():
    """
    Applies compatibility patches for PyTorch 2.10+ / CUDA 12.8 envs
    where older libraries (pyannote, whisperx) expect deprecated APIs.
    """
    # 0. Silence noisy deprecation warnings from libraries we cannot control
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
    warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
    
    # 1. Revert to pre-PyTorch 2.6 behavior for loading checkpoints.
    # This bypasses the weights_only=True requirement globally for this process.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    # 2. Restore torchaudio.AudioMetaData (Removed in 2.10, needed by pyannote < 3.3)
    try:
        torchaudio.AudioMetaData
    except AttributeError:
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding
        torchaudio.AudioMetaData = AudioMetaData

    # 3. Restore list_audio_backends (Deprecated/Removed)
    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            return ["soundfile", "ffmpeg"]
        torchaudio.list_audio_backends = list_audio_backends

    # 4. Restore get_audio_backend (Deprecated/Removed)
    if not hasattr(torchaudio, "get_audio_backend"):
        def get_audio_backend():
            return "soundfile"
        torchaudio.get_audio_backend = get_audio_backend

    # 6. Restore set_audio_backend (Removed in 2.2+, used by pyannote < 4.0)
    if not hasattr(torchaudio, "set_audio_backend"):
        def set_audio_backend(backend):
            pass # No-op in newer torchaudio
        torchaudio.set_audio_backend = set_audio_backend

