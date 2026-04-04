import subprocess
import numpy as np
import soundfile as sf
import librosa 
import os 
import sys

from logger import logger
from models.base import BaseModel
try:
    import opensmile
except ImportError:
    logger.error("Please install opensmile: pip install opensmile")
    raise



class OpenSmileExtractor(BaseModel):
    def __init__(self, 
                 model_name: str="opensmile" ,
                 model_path: str="opensmile", 
                 feature_list: list=[],
                 gpu_id: int=None,
                 gpu_thread_id: int=None,
                 device: str="cpu",
                 sampling_rate: int=16000,
                 open_smile_config: dict={},
                 ):
        
        super().__init__(model_name, model_path, feature_list, device)
        self.sampling_rate = sampling_rate
        self.gpu_id = gpu_id
        self.gpu_thread_id = gpu_thread_id
        self.open_smile_config = open_smile_config
        self.feature_set = self.open_smile_config.get("feature_set", ["eGeMAPSv02", "emobase"])
        self.feature_levels = self.open_smile_config.get("feature_levels", ["LowLevelDescriptors"])
        self.smiles = {}
        for feature in self.feature_set:
            for feature_level in self.feature_levels:
                self.smiles[f"{feature}_{feature_level}"] = opensmile.Smile(
                    feature_set=getattr(opensmile.FeatureSet, feature),
                    feature_level=getattr(opensmile.FeatureLevel, feature_level),
                )
                logger.info(f"Initialized OpenSmile extractor for feature set {feature} and level {feature_level}.")
    
    def extract_opensmile_features(self, audio):
        outputs = {"audio_shape": audio.shape}

        for feature in self.feature_set:
            for feature_level in self.feature_levels:
                key = f"{feature}_{feature_level}"
                smile = self.smiles[key]
                smile_features = smile.process_signal(audio, self.sampling_rate)  # pandas DataFrame
                column_names = smile_features.columns
                
                for col in column_names:
                    outputs[f"{key}_{col}"] = smile_features[col].astype(np.float32)

                # 2) index(start/end) ]
                if hasattr(smile_features.index, "names") and len(smile_features.index.names) == 2:
                    start_list = smile_features.index.get_level_values(0).astype(str)
                    end_list = smile_features.index.get_level_values(1).astype(str)
                else:
                    start_list, end_list = [], []
                outputs[f"{key}_column_names"] = column_names
                # outputs[f"{key}_start"] = start_list
                # outputs[f"{key}_end"] = end_list

        return outputs

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
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", audio_path_wav], check=True, capture_output=True)
            audio, sr = sf.read(audio_path_wav)
            os.remove(audio_path_wav)
        audio = audio.astype("float32")

        if sr != self.sampling_rate:
            # resample the audio to desired 16kHz sampling rate
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sampling_rate).T
        return audio

    def extract_features(self, audio_path):
        audio_data = self.load_audio(audio_path)

        if "open_smile_features" in self.feature_list:
            opensmile_features = self.extract_opensmile_features(audio_data)
            return [("open_smile_features", opensmile_features)]
        
if __name__ == "__main__":
    extractor = OpenSmileExtractor(model_path="opensmile", device="cuda")
    features = extractor.extract_features("audio.wav")
    for feature_name, feature_value in features:
        print(f"{feature_name}: {feature_value['mels'].shape}")