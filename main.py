import argparse
import queue
import threading

from config import read_config
from database import init_db
from logger import logger
from utils import list_all_videos
from worker import gpu_worker_thread, db_write_thread

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--config_path", type=str, default="config/navit-config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config_path
    
    # load the configuration
    conf = read_config(config_path)
    video_paths = list_all_videos(conf["video_input"]["path"])
    logger.info(f"Found {len(video_paths)} videos.")

    # Queues
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # Enqueue all video paths
    for path in video_paths:
        task_queue.put(path)

    # Initialize DB
    db = init_db(conf["db"])
    
    # # Start DB writing thread
    writer = threading.Thread(target=db_write_thread, args=(result_queue, db))
    writer.start()

    # Start GPU worker threads
    gpu_threads = []
    for gpu_id in conf["gpus"]:
        t = threading.Thread(target=gpu_worker_thread, args=(gpu_id, task_queue, result_queue, conf["model"]))
        t.start()
        gpu_threads.append(t)

    # # Wait for all tasks to be processed
    for t in gpu_threads:
        t.join()

    # Wait for the video tasks to finish
    task_queue.join()

    # Wait for the DB writer to finish
    result_queue.put(None)
    result_queue.join()
    writer.join()

    logger.success("✅ All tasks completed.")