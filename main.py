import argparse
import queue
import threading
import time

from config import read_config
from database import init_db
from logger import logger
from utils import list_all_videos
from worker import gpu_worker_thread, db_write_thread

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--config_path", type=str, default="config/sample-test.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config_path
    
    # load the configuration
    conf = read_config(config_path)
    video_paths = list_all_videos(conf["video_input"]["path"])
    logger.info(f"Found {len(video_paths)} videos.")

    try:
        for phase in conf["phases"]:
            # Queues
            task_queue = queue.Queue()
            result_queue = queue.Queue()

            # Enqueue all video paths
            for path in video_paths:
                task_queue.put(path)

            stop_event = threading.Event()

            # # Start DB writing thread
            writer_threads = []
            db = init_db(phase["db"])
            for writer_thread_id in range(conf["writers"]):
                w = threading.Thread(target=db_write_thread, args=(result_queue, writer_thread_id, db))
                w.start()
                writer_threads.append(w)

            # Start GPU worker threads
            gpu_threads = []
            for gpu_id in conf["gpus"]:
                for gpu_thread_id in range(conf["threads_per_gpu"]):
                    time.sleep(5)
                    t = threading.Thread(target=gpu_worker_thread, args=(gpu_id, gpu_thread_id, task_queue, result_queue, phase["model"], stop_event))
                    t.start()
                    gpu_threads.append(t)

            # # Wait for all tasks to be processed
            for t in gpu_threads:
                t.join()

            # Wait for the DB writer to finish
            for _ in range(conf["writers"]):
                result_queue.put(None)
            for w in writer_threads:
                w.join()
            db.close()

    except KeyboardInterrupt:
        logger.warning("⚠️ Ctrl+C received! Shutting down gracefully...")
        stop_event.set()

        # Clear the task queue
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for GPU threads
        for t in gpu_threads:
            t.join()

        # Signal writers
        for _ in range(conf["writers"]):
            result_queue.put(None)
        for w in writer_threads:
            w.join()

        db.close()
        logger.warning("Graceful shutdown complete.")

    logger.success("✅ All tasks completed.")