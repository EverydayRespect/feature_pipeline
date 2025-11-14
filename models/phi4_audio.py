# Extracting mel-spectrograms for Phi-4 audio/speech related tasks
import numpy as np
import soundfile as sf
import librosa 
import os 

from logger import logger
from models.base import BaseModel

# Reference: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L260
def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.
    Args:
        sample_rate (int): Sample rate in Hz. number > 0 [scalar]
        n_fft (int): FFT size. int > 0 [scalar]
        n_mel (int): Mel filter size. int > 0 [scalar]
        fmin (float): lowest frequency (in Hz). If None use 0.0.
            float >= 0 [scalar]
        fmax: highest frequency (in Hz). If None use sample_rate / 2.
            float >= 0 [scalar]
    Returns
        out (numpy.ndarray): Mel transform matrix
            [shape=(n_mels, 1 + n_fft/2)]
    """

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    print(f"fmin: {fmin} fmax: {fmax} sample_rate: {sample_rate}")
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix

class Phi4MelExtractor(BaseModel):
    def __init__(self, model_name: str=None, 
                 model_path: str=None, 
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
        
        self.mel_filters = speechlib_mel(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=None,
            fmax=7690
        ).T
        self.window = np.hamming(win_length)
        self.preemphasis = 0.97
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_mels(self, audio, output="mel_features"):
        n_batch = (audio.shape[0] - self.win_length)//self.hop_length + 1
        y_frames = np.array(
                    [audio[_stride : _stride + self.win_length] for _stride in range(0, self.hop_length * n_batch, self.hop_length)],
                    dtype=np.float32
                    )
        
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - self.preemphasis * y_frames_prev) * 32678

        S = np.fft.rfft(self.window * y_frames, n=self.n_fft, axis=1).astype(np.complex64)
        spec = np.abs(S).astype(np.float32)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self.mel_filters), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        if output == "mel_features":
            return {
                "mels": log_fbank,
                "audio_shape": audio.shape
            }
    
    def load_audio(self, audio_path):
        if audio_path.endswith(".wav"):
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        elif audio_path.endswith(".mp4"):
            audio_path_wav = audio_path.replace(".mp4", ".wav")
            os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {audio_path_wav}")
            audio, sr = sf.read(audio_path_wav)
            os.system(f"rm {audio_path_wav}")
        audio = audio.astype("float32")

        if sr != 16000:
            # resample the audio to desired 16kHz sampling rate
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=16000).T
        return audio

    def extract_features(self, audio_path):
        audio_data = self.load_audio(audio_path)

        if "speech_mels_features" in self.feature_list:
            mel_features = self.extract_mels(audio_data)
            return [("speech_mels_features", mel_features)]


        
