import numpy as np
import soundfile as sf
import librosa 
import os 
import sys

import torch
import torchaudio.compliance.kaldi as ta_kaldi
from logger import logger

from models.base import BaseModel

class BEATsExtractor(BaseModel):
    def __init__(self, model_name: str="BEATs" , 
                 model_path: str="", 
                 feature_list: list=[], 
                 gpu_id: int=None,
                 gpu_thread_id: int=None,
                 device: str="cpu",
                 n_mels: int = 128,
                 sampling_rate: int=16000,
                 frame_length: int = 25,
                 frame_shift: int = 10,
                 fbank_mean: float = 15.41663,
                 fbank_std: float = 6.55582,):
        
        super().__init__(model_name, model_path, feature_list, device)
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
    
    def extract_mels(self, audio, output="mel_features"):
        audio = torch.from_numpy(audio).unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(
            audio,
            num_mel_bins=self.n_mels,
            sample_frequency=self.sampling_rate,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )

        fbank = (fbank - self.fbank_mean) / self.fbank_std

        if output == "mel_features":
            return {
                "mels": fbank.numpy(),
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
        