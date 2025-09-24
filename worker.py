import queue
import time
import torch
from logger import logger
from models.utils import load_model

def gpu_worker_thread(gpu_id, gpu_thread_id, task_queue, data_queue, model_conf, stop_event):
    """
    GPU worker thread: extract embeddings from videos and push results into data_queue
    for the writer process.
    """

    # Load model on this GPU
    try:
        extractor = load_model(gpu_id, gpu_thread_id, model_conf)
    except Exception as e:
        logger.error(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Failed to load model: {e}")
        logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Worker exiting.")
        return 
    
    while not stop_event.is_set():
        # Fetch a video path to process
        try:
            video_path = task_queue.get(timeout=2)
        except queue.Empty:
            logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] No more tasks. Exited.")
            return

        logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Start processing {video_path}...")

        try:
            # Iterate over extracted features (frame embeddings)
            for feature_name, feature_value in extractor.extract_features(video_path):
                fid  = feature_value["frame_id"]
                rows = feature_value["grid_rows"]
                cols = feature_value["grid_cols"]
                embs = feature_value["embeddings"]

                # Ensure embeddings are on CPU and numpy
                print(embs.shape)
                if isinstance(embs, torch.Tensor):
                    embs = embs.to(torch.float32).numpy()

                record = {
                    "type": feature_name,
                    "video_path": video_path,
                    "frame_id": fid,
                    "grid_rows": rows,
                    "grid_cols": cols,
                    "embeddings": embs,
                }

                # Push record into writer queue
                data_queue.put(record)
                
            data_queue.put({
                "type": "video_done",
                "video_path": video_path
            })

        except Exception as e:
            logger.error(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Error processing {video_path}: {e}")
        finally:
            logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Finished {video_path}")

    logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Exit signal received. Exited.")
