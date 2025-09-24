import time
import numpy as np
import ffmpeg
import av
import cv2

VIDEO_PATH = "../../video_samples/atw.mp4"  # 替换为你的15min视频路径
FPS = 1  # 采样频率，可改成 None 表示解码全部

def benchmark_ffmpeg(video_path, fps=FPS):
    start = time.time()
    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    w, h = int(video_stream['width']), int(video_stream['height'])

    stream = ffmpeg.input(video_path)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=True)

    frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    end = time.time()
    return len(frames), end - start


def benchmark_pyav(video_path, fps=FPS):
    start = time.time()
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if fps is not None and i % int(stream.average_rate / fps) != 0:
            continue
        frames.append(frame.to_rgb().to_ndarray())
    end = time.time()
    return len(frames), end - start


def benchmark_opencv(video_path, fps=FPS):
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    frames = []
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(native_fps / fps) if fps else 1
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fps and i % step != 0:
            i += 1
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    end = time.time()
    return len(frames), end - start


if __name__ == "__main__":
    print("Benchmarking on:", VIDEO_PATH)

    n, t = benchmark_ffmpeg(VIDEO_PATH)
    print(f"ffmpeg-python: {n} frames, {t:.2f} sec, {n/t:.2f} FPS")

    n, t = benchmark_pyav(VIDEO_PATH)
    print(f"PyAV:          {n} frames, {t:.2f} sec, {n/t:.2f} FPS")

    n, t = benchmark_opencv(VIDEO_PATH)
    print(f"OpenCV:        {n} frames, {t:.2f} sec, {n/t:.2f} FPS")
