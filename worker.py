import queue
import time
from database import BaseVectorDB, init_db
from logger import logger
from models.utils import load_model
from file_writer import ParquetShardWriter

def build_frame_records(video_path: str, feature_name: str, feature_value: dict):
    """
    将一帧的 embedding 打成一条记录：
    - embeddings: 形状可为 (rows*cols, dim) 或 (rows, cols, dim) 或 (dim,)。最终统一为 (num_patches, dim)
    - 主键：frame_id（配合 video_path 保证唯一）
    """
    fid  = feature_value["frame_id"]
    rows = feature_value["grid_rows"]
    cols = feature_value["grid_cols"]
    embs = feature_value["embeddings"]  

    rec = {
        "video_path": video_path,
        "feature_name": feature_name,
        "frame_id": fid,          
        "grid_rows": rows,
        "grid_cols": cols,
        "embeddings": embs,       
    }
    return [rec]                  

def gpu_worker_thread(gpu_id, gpu_thread_id, task_queue, writer: ParquetShardWriter, model_conf, stop_event):
    
    # Load model onto device
    try:
        extractor = load_model(gpu_id, gpu_thread_id, model_conf)
    except Exception as e:
        logger.error(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Failed to load model: {e}")
        logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Worker exiting due to model load failure.")
        return 
    
    while not stop_event.is_set():
        # Get next video to process
        try:
            video_path = task_queue.get(timeout=2)
        except queue.Empty:
            logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] No more tasks. Exited.")
            return

        logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Start processing {video_path}...")

        try:
            # Get features and enqueue to writer
            for feature_name, feature_value in extractor.extract_features(video_path):
                records = build_frame_records(video_path, feature_name, feature_value)
                writer.enqueue(records)
        except Exception as e:
            logger.error(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Error processing {video_path}: {e}")
        finally:
            logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Finished processing {video_path}...")
            task_queue.task_done()

    logger.info(f'[GPU-{gpu_id}-Thread-{gpu_thread_id}] Exit signal recieved. Exited.')

def db_write_thread(result_queue, writer_thread_id, db: BaseVectorDB):
    while True:
        try:
            data = result_queue.get()
            if data is None:
                # Signal to exit
                result_queue.task_done()
                break

            # logger.info(f"[DB Write Thread {writer_thread_id}] Writing {data['video_path']}, {data['feature_name']} to DB...")
            db.insert(data['feature_name'], data['video_path'], data['feature_value'])
            logger.info(f"[DB Write Thread {writer_thread_id}] Successfully wrote {data['video_path']}, {data['feature_name']} to DB.")
            result_queue.task_done()
        except Exception as e:
            logger.error(f"[DB Write Thread {writer_thread_id}] Error while inserting into DB: {e}")
            result_queue.task_done()