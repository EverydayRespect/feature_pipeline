import argparse
import queue
import threading
import time
import os

from config import read_config
from logger import logger
from utils import list_all_videos
# Make sure gpu_worker_thread signature is: (gpu_id, gpu_thread_id, task_queue, writer, model_conf, stop_event)
from worker import gpu_worker_thread
from file_writer import ParquetShardWriter  # the per-worker parquet shard writer we discussed

def launch_workers_for_phase(conf, phase, video_paths):
    """
    Launch task queue, per-worker writers, and GPU worker threads for one phase.
    Returns (task_queue, writers_dict, gpu_threads, stop_event)
    """
    # Queues
    task_queue = queue.Queue()

    # Enqueue tasks
    for p in video_paths:
        task_queue.put(p)

    stop_event = threading.Event()

    # Writer settings (with sensible defaults)
    buffer_root = conf.get("buffer_root", "./buffer")
    groups      = int(conf.get("groups", 1))
    writer_cfg  = conf.get("writer", {})
    shard_rows        = int(writer_cfg.get("shard_rows", 20_000))
    max_queue         = int(writer_cfg.get("max_queue", 5_000))
    compression       = writer_cfg.get("compression", "zstd")
    embedding_dtype   = writer_cfg.get("embedding_dtype", "float32")
    flush_interval_s  = float(writer_cfg.get("flush_interval_s", 60.0))

    os.makedirs(buffer_root, exist_ok=True)

    # Create per-worker writers and GPU worker threads
    writers = {}
    gpu_threads = []
    gpus = conf["gpus"]
    threads_per_gpu = int(conf["threads_per_gpu"])

    for gpu_id, device in enumerate(gpus):
        for gpu_thread_id in range(threads_per_gpu):
            group_id = (gpu_id * threads_per_gpu + gpu_thread_id) % groups
            group_root = os.path.join(buffer_root, conf.get("video_input").get("batch"), "group"+str(group_id))
            os.makedirs(group_root, exist_ok=True)

            writer = ParquetShardWriter(
                root_dir=group_root,
                group_id=group_id,
                gpu_id=gpu_id,
                worker_id=gpu_thread_id,
                shard_rows=shard_rows,
                max_queue=max_queue,
                compression=compression,
                embedding_dtype=embedding_dtype,
                flush_interval_s=flush_interval_s,
            )
            writers[(gpu_id, gpu_thread_id)] = writer

            t = threading.Thread(
                target=gpu_worker_thread,
                args=(device, gpu_thread_id, task_queue, writer, phase["model"], stop_event),
                name=f"{device}-t{gpu_thread_id}",
                daemon=True,
            )
            t.start()
            gpu_threads.append(t)

    return task_queue, writers, gpu_threads, stop_event


def shutdown_workers(task_queue, writers, gpu_threads, stop_event, reason="normal"):
    """
    Gracefully stop workers and flush/close writers.
    """
    logger.info(f"Shutting down ({reason})…")

    # Stop GPU worker loops
    stop_event.set()

    # Drain task queue (so workers can exit quickly)
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
        except queue.Empty:
            break

    # Join GPU worker threads
    for t in gpu_threads:
        t.join()

    # Close writers (flush remaining buffers and stop writer threads)
    for w in writers.values():
        try:
            w.close()
        except Exception as e:
            logger.error(f"Writer close error: {e}")

    logger.success("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--config_path", type=str, default="config/sample-test.yaml",
                        help="Path to the configuration file")
    args = parser.parse_args()

    conf = read_config(args.config_path)
    video_paths = list_all_videos(conf["video_input"]["path"])
    logger.info(f"Found {len(video_paths)} videos.")

    try:
        for phase in conf["phases"]:
            logger.info(f"Starting phase: {phase.get('name', '<unnamed>')}")
            task_queue, writers, gpu_threads, stop_event = launch_workers_for_phase(conf, phase, video_paths)

            # Wait for all GPU workers to finish this phase
            for t in gpu_threads:
                t.join()

            # Close writers after workers finish
            shutdown_workers(task_queue, writers, gpu_threads, stop_event, reason="phase complete")

        logger.success("✅ All tasks completed.")

    except KeyboardInterrupt:
        # Best-effort graceful shutdown for the current phase
        logger.warning("⚠️ Ctrl+C received! Shutting down gracefully…")
        try:
            shutdown_workers(task_queue, writers, gpu_threads, stop_event, reason="KeyboardInterrupt")
        except Exception:
            # If we were interrupted before these are defined (e.g. early Ctrl+C)
            pass
        logger.warning("Graceful shutdown complete.")
