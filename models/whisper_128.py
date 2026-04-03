import numpy as np
import soundfile as sf
import librosa 
import os 
import sys

from logger import logger
from models.base import BaseModel
try:
    import whisper
except :
    logger.error("Please install whisper: pip install openai-whisper")
    raise



class Whisper128Extractor(BaseModel):
    def __init__(self, 
                 model_name: str="Whisper128" ,
                 model_path: str="models/Whisper128", 
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
        self.chunk_length = 30 * self.sampling_rate  # 30 seconds
        self.n_mels = 128

        # load audio and pad/trim it to fit 30 seconds
        # make log-Mel spectrogram and move to the same device as the model
    def extract_mels(self, audio, output="mel_features"):
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        if output == "mel_features":
            
            return {
                "mels": mel.cpu().numpy() ,  # shape [num_patches, hidden_dim]
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
        audio = whisper.load_audio(audio_path, sr=self.sampling_rate)
        return audio

    def extract_features(self, audio_path):
        audio_data = self.load_audio(audio_path)

        if "speech_mels_features" in self.feature_list:
            mel_features = self.extract_mels(audio_data)
            return [("speech_mels_features", mel_features)]
        
if __name__ == "__main__":
    extractor = Whisper128Extractor(model_path="models/Whisper128", device="cuda")
    features = extractor.extract_features("audio.wav")
    for feature_name, feature_value in features:
        print(f"{feature_name}: {feature_value['mels'].shape}")