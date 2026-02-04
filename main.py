import argparse
import multiprocessing as mp
import threading
import os

from config import read_config
from logger import logger
from utils import list_all_videos
from worker import gpu_worker_thread
from file_writer_arrow import ArrowWriterProcess   # 👈 new writer process

def launch_workers_for_phase(conf, phase, video_paths):
    """
    Launch the task queue, a single writer process, and GPU worker threads.
    Returns (task_queue, gpu_threads, writer_proc, data_queue, stop_event)
    """
    # Task queue (video paths to process)
    task_queue = mp.Queue()
    for p in video_paths:
        task_queue.put(p)

    # Data queue (GPU results → writer process)
    data_queue = mp.Queue(maxsize=10000)

    stop_event = mp.Event()

    # Start the writer process
    buffer_root = os.path.join(conf.get("buffer_root", "/mnt/14t_drive"), phase["model"]["base_dir"])
    os.makedirs(buffer_root, exist_ok=True)
    writer_proc = mp.Process(
        target=ArrowWriterProcess,
        args=(buffer_root, data_queue, stop_event),
        daemon=True,
    )
    writer_proc.start()

    # Start GPU worker threads
    gpu_threads = []
    gpus = conf["gpus"]
    threads_per_gpu = int(conf["threads_per_gpu"])
    for gpu_id, device in enumerate(gpus):
        for gpu_thread_id in range(threads_per_gpu):
            t = threading.Thread(
                target=gpu_worker_thread,
                args=(device, gpu_thread_id, task_queue, data_queue, phase["model"], stop_event),
                name=f"{device}-t{gpu_thread_id}",
                daemon=True,
            )
            t.start()
            gpu_threads.append(t)

    return task_queue, gpu_threads, writer_proc, data_queue, stop_event


def shutdown_workers(task_queue, gpu_threads, writer_proc, data_queue, stop_event, reason="normal"):
    """
    Gracefully stop GPU workers and the writer process.
    """
    logger.info(f"Shutting down ({reason})…")

    # Stop GPU workers
    stop_event.set()
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
        except Exception:
            break
    for t in gpu_threads:
        t.join()

    # Stop the writer process
    data_queue.put(None)   # termination signal
    writer_proc.join()

    logger.success("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--config_path", type=str, default="config/first_batch_pyarrow.yaml",
                        help="Path to the configuration file")
    args = parser.parse_args()

    conf = read_config(args.config_path)
    if conf["video_input"].get("input_path", None):
        with open(conf["video_input"].get("input_path", None)) as f:
            video_paths = [line.strip() for line in f]
    else:
        video_paths = list_all_videos(conf["video_input"]["path"])
    
    logger.info(f"Found {len(video_paths)} videos.")
    logger.info(video_paths)

    # Initialize worker-related variables so they are always defined if an exception occurs early
    task_queue = None
    gpu_threads = []
    writer_proc = None
    data_queue = None
    stop_event = None

    try:
        for phase in conf["phases"]:
            logger.info(f"Starting phase: {phase.get('name', '<unnamed>')}")
            task_queue, gpu_threads, writer_proc, data_queue, stop_event = launch_workers_for_phase(conf, phase, video_paths)

            # Wait for GPU workers to finish
            for t in gpu_threads:
                t.join()

            # Shut everything down
            shutdown_workers(task_queue, gpu_threads, writer_proc, data_queue, stop_event, reason="phase complete")

        logger.success("✅ All tasks completed.")

    except KeyboardInterrupt:
        # Best-effort graceful shutdown
        logger.warning("⚠️ Ctrl+C received! Shutting down gracefully…")
        shutdown_workers(task_queue, gpu_threads, writer_proc, data_queue, stop_event, reason="KeyboardInterrupt")
        logger.warning("Graceful shutdown complete.")
