import numpy as np
import soundfile as sf
import librosa 
import os 
import sys

from logger import logger
from models.base import BaseModel

try:
    from transformers import WavLMModel, AutoFeatureExtractor
except ImportError:
    logger.error("Please install transformers: pip install transformers")
    raise

class WavLMExtractor(BaseModel):
    def __init__(self, model_name: str="microsoft/wavlm-large" , 
                 model_path: str="./WavLM-Large", 
                 feature_list: list=[], 
                 gpu_id: int=None,
                 gpu_thread_id: int=None,
                 device: str="cpu",
                 sampling_rate: int=16000,
                 n_mels: int=80,
                 hop_length: int=160,
                 n_fft: int=512,
                 win_length: int=400,
                 dither: float=0.0,
                 padding_value: float=0.0):
        
        super().__init__(model_name, model_path, feature_list, device)
        
        # Use local dir only if it exists, otherwise fallback to HF model name
        load_path = os.path.abspath(model_path) if (model_path and os.path.isdir(model_path)) else model_name
        logger.info(f"Loading WavLM model from {load_path}...")
        
        self.processor = AutoFeatureExtractor.from_pretrained(load_path)
        self.feature_extractor = WavLMModel.from_pretrained(
            load_path
        ).feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.dither = dither
        self.padding_value = padding_value
        self.chunk_length = 30 * self.sampling_rate  # 30 seconds in samples
    
    def extract_mels(self, audio, output="mel_features"):
        # 1) chunk audio (keep last short chunk)
        audio_chunks = []
        for start in range(0, len(audio), self.chunk_length):
            end = start + self.chunk_length
            chunk = audio[start:end]
            if chunk.shape[0] > 0:
                audio_chunks.append(chunk)

        if len(audio_chunks) == 0:
            raise ValueError("Empty audio after chunking.")

        # 2) extract per-chunk features
        chunk_features = []
        for chunk in audio_chunks:
            inputs = self.processor(
                chunk,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )["input_values"].to(self.device)

            feat = self.feature_extractor(inputs)  # [B, D, T]
            chunk_features.append(feat.detach().cpu().numpy().squeeze(0))  # [D, T]

        # 3) concat on time axis (last short chunk included)
        features = np.concatenate(chunk_features, axis=-1)

        if output == "mel_features":
            return {
                "mels": features,
                "audio_shape": audio.shape
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
    extractor = WavLMExtractor(model_path="./WavLM-Large", device="cuda")
    features = extractor.extract_features("audio.wav")
    for feature_name, feature_value in features:
        print(f"{feature_name}: {feature_value['mels'].shape}")