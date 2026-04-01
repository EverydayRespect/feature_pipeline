import numpy as np
import soundfile as sf
import librosa 
import os 
import sys

from logger import logger
from models.base import BaseModel
try:
    from transformers import AutoProcessor
except ImportError:
    logger.error("Please install transformers: pip install transformers")
    raise

try:
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRProcessor,
    )
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
except ImportError:
    logger.error("Please install qwen_asr: pip install -U qwen-asr")
    raise


class Qwen3ASRExtractor(BaseModel):
    def __init__(self, 
                 model_name: str="Qwen/Qwen3-ASR-0.6B" ,
                 model_path: str="models/Qwen3-ASR-0.6B", 
                 feature_list: list=[],
                 gpu_id: int=None,
                 gpu_thread_id: int=None,
                 device: str="cpu",
                 sampling_rate: int=16000,
                 ):
        
        super().__init__(model_name, model_path, feature_list, device)
        self.sampling_rate = sampling_rate
        self.gpu_id = gpu_id
        self.gpu_thread_id = gpu_thread_id
        # Use model_path if provided (downloaded local directory), 
        # otherwise use model_name (Hugging Face model name)
        load_path = os.path.abspath(model_path) if os.path.abspath(model_path) else model_name
        logger.info(f"Loading Qwen3ASR model from {load_path}...")
        
        self.processor = AutoProcessor.from_pretrained(load_path, fix_mistral_regex=True)
        self.prompt = ['<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n']
    
    def extract_mels(self, audio, output="mel_features"):

        outputs = self.processor(text=self.prompt, audio=[audio], return_tensors="pt", padding=True)

        if output == "mel_features":
            
            return {
                "mels": outputs.input_features.cpu().numpy() ,  # shape [num_patches, hidden_dim]
                "audio_shape": audio.shape,  # original audio shape
            }

    def load_audio(self, audio_path: str = None,):
        """
        Load an audio file and resample if necessary.

        Args:
            audiopath (str, optional): Path to the input audio file.

        Returns:
            torch.Tensor: A tuple containing the audio tensor and the sample rate.
        """
        if audio_path.endswith(".wav"):
            audio, sr = sf.read(audio_path)
            logger.info(f"Loaded audio {audio_path} with shape {audio.shape} and original SR {sr}.")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            logger.info(f"Loaded audio {audio_path} with shape {audio.shape} and original SR {sr}.")
        elif audio_path.endswith(".mp4"):
            audio_path_wav = audio_path.replace(".mp4", ".wav")
            os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {audio_path_wav}")
            audio, sr = sf.read(audio_path_wav)
            os.system(f"rm {audio_path_wav}")
        audio = audio.astype("float32")

        if sr != self.sampling_rate:
            # resample the audio to desired 16kHz sampling rate
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sampling_rate).T
        return audio

    def extract_features(self, audio_path):
        audio_data = self.load_audio(audio_path)

        if "speech_mels_features" in self.feature_list:
            mel_features = self.extract_mels(audio_data)
            return [("speech_mels_features", mel_features)]
        
if __name__ == "__main__":
    extractor = Qwen3ASRExtractor(model_path="models/Qwen3-ASR-0.6B", device="cuda")
    features = extractor.extract_features("audio.wav")
    for feature_name, feature_value in features:
        print(f"{feature_name}: {feature_value['mels'].shape}")