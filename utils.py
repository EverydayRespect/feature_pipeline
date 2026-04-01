from typing import Optional, Optional, List, Tuple
import ffmpeg
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import time

import soundfile as sf 
import librosa
from lhotse import Recording, CutSet

MAX_FRAMES = 768

def list_all_videos(input_paths, exts=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Recursively list all video files in a directory.

    Args:
        input_path (str): Root directory to search.
        exts (tuple): Video file extensions to include.

    Returns:
        List[str]: List of full paths to video files.
    """
    video_paths = []
    for input_path in input_paths:
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(exts):
                    full_path = os.path.join(root, file)
                    video_paths.append(full_path)
    return video_paths

def load_audio(audio_path: str,
               sampling_rate: int=16000,
               backend: str="soundfile"):
    """
    Load an audio file and (optionally) resample it to a target sampling rate.

    This function supports two backends for audio loading:
    1. **soundfile** – Uses the `soundfile` library to read audio data and resamples
       it with `librosa.resample` if the original sample rate differs from the target.
    2. **cutset** – Uses a `CutSet` and `Recording` interface (from Lhotse) to load
       and resample audio, returning both the waveform and its lengths.

    Args:
        audio_path (str): Path to the audio file to load.
        sampling_rate (int, optional): Desired sampling rate in Hz. 
            Defaults to 16000.
        backend (str, optional): Audio loading backend to use.
            Choices are `"soundfile"` or `"cutset"`. Defaults to `"soundfile"`.

    Returns:
        tuple:
            - For `"soundfile"` backend:
                (np.ndarray, int): The audio waveform and the (possibly resampled) sampling rate.
            - For `"cutset"` backend:
                (torch.Tensor, torch.Tensor, int): The audio tensor, 
                corresponding audio lengths, and the sampling rate.

    Example:
        >>> audio, sr = load_audio("example.wav", sampling_rate=16000)
        >>> audio.shape
        (16000,)
    """
    if backend == "soundfile":
        audio, audio_sampling_rate = sf.read(audio_path)
        if sampling_rate != audio_sampling_rate:
            # resample the audio to desired sampling_rate
            audio = librosa.resample(audio.T, orig_sr=audio_sampling_rate, target_sr=sampling_rate).T
        return audio, sampling_rate
    elif backend == "cutset":
        cuts = CutSet([Recording.from_file(audio_path).to_cut()])
        audio, audio_lens = cuts.resample(sampling_rate).load_audio(collate=True)
        return audio, audio_lens, sampling_rate

def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = 2,
    max_frames: Optional[float] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    precise_time: bool = False,
    verbose: bool = False,
    temporal_factor: int = 1,
):
    """
    Load and process a video file and return the frames and the timestamps of each frame.

    Args:
        video_path (str): Path to the video file.
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.
        fps (float, optional): Frames per second. Defaults to None.
        num_frames (float, optional): Number of frames to sample. Defaults to None.
        size (int, optional): Size of the shortest side. Defaults to None.
        size_divisible (int, optional): Size divisible by this number. Defaults to 1.
        precise_time (bool, optional): Whether to use precise time. Defaults to False.
        verbose (bool, optional): Print ffmpeg output. Defaults to False.

    Returns:
        frames (List[PIL.Image]): List of frames.
        timestamps (List[float]): List of timestamps.
    """
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print("STDERR from ffprobe:")
        print(e.stderr.decode())   # 🔥 打印 ffprobe stderr
        raise
    duration = float(probe['format']['duration'])
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    w, h = int(video_stream['width']), int(video_stream['height'])

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream['start_time']), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream['start_time'])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {'ss': start_time, 't': duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    if size is not None:
        scale_factor = size / min(w, h)
        new_w, new_h = round(w * scale_factor), round(h * scale_factor)
    else:
        new_w, new_h = w, h
    new_w = new_w // size_divisible * size_divisible
    new_h = new_h // size_divisible * size_divisible

    # NOTE: It may result in unexpected number of frames in ffmpeg
    # if calculate the fps directly according to max_frames
    # NOTE: the below lines may hurt the performance
    # if max_frames is not None and (fps is None or duration * fps > 2 * max_frames):
    #     fps = max_frames / duration * 2

    stream = ffmpeg.input(video_path, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if new_w != w or new_h != h:
        stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[:len(frames)]
    else:
        timestamps = np.linspace(start_time, start_time + duration, len(frames))

    # Limit the number of frames to max_frames if specified
    # max_frames = max_frames if max_frames is not None else MAX_FRAMES
    # if max_frames is not None and len(frames) > max_frames:
    #     indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    #     frames = frames[indices]
    #     timestamps = [timestamps[i] for i in indices]

    # Pad the frames to be divisible by temporal_factor
    # if temporal_factor > 1:
    #     pad_length = temporal_factor - len(frames) % temporal_factor
    #     frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
    #     [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    frames = [frame for frame in frames]

    return frames, timestamps

def load_video_frames(
    video_path: str,
    fps: float = 2,
    start_time: float = None,
    end_time: float = None,
    verbose: bool = False,
):
    """
    Load video and return:
        frames: List[PIL.Image]
        timestamps: List[float]
    """

    # -----------------------------------
    # 1️⃣ Probe video
    # -----------------------------------
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )

    w, h = int(video_stream["width"]), int(video_stream["height"])
    duration = float(probe["format"]["duration"])
    start_time_video = float(video_stream.get("start_time", 0.0))

    # -----------------------------------
    # 2️⃣ Trim logic
    # -----------------------------------
    input_kwargs = {}

    if start_time is not None:
        input_kwargs["ss"] = start_time

    if end_time is not None:
        if start_time is not None:
            input_kwargs["t"] = end_time - start_time
        else:
            input_kwargs["t"] = end_time - start_time_video

    # -----------------------------------
    # 3️⃣ Decode raw RGB
    # -----------------------------------
    stream = (
        ffmpeg
        .input(video_path, **input_kwargs)
        .filter("fps", fps=fps)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
    )

    out, _ = ffmpeg.run(
        stream,
        capture_stdout=True,
        quiet=not verbose
    )

    # -----------------------------------
    # 4️⃣ Convert to numpy (HWC)
    # -----------------------------------
    frame_size = w * h * 3
    num_frames = len(out) // frame_size

    frames_np = (
        np.frombuffer(out, np.uint8)
        .reshape(num_frames, h, w, 3)
    )

    # -----------------------------------
    # 5️⃣ Convert to PIL
    # -----------------------------------
    frames = [
        Image.fromarray(frame)
        for frame in frames_np
    ]

    return frames, _

