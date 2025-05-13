from typing import Optional
import ffmpeg
import numpy as np
import os

MAX_FRAMES = 768

def list_all_videos(input_path, exts=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Recursively list all video files in a directory.

    Args:
        input_path (str): Root directory to search.
        exts (tuple): Video file extensions to include.

    Returns:
        List[str]: List of full paths to video files.
    """
    video_paths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)
    return video_paths

def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = 1,
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
    probe = ffmpeg.probe(video_path)
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
