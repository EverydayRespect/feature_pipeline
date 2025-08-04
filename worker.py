import queue
from database import BaseVectorDB, init_db
from logger import logger
from models.utils import load_model

def gpu_worker_thread(gpu_id, gpu_thread_id, task_queue, result_queue, model_conf, stop_event):
    
    try:
        extractor = load_model(gpu_id, gpu_thread_id, model_conf)
    except Exception as e:
        logger.error(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Failed to load model: {e}")
        logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Worker exiting due to model load failure.")
        return 
    
    while not stop_event.is_set():
        try:
            video_path = task_queue.get(timeout=2)
        except queue.Empty:
            logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] No more tasks. Exited.")
            return

        logger.info(f"[GPU-{gpu_id}-Thread-{gpu_thread_id}] Start processing {video_path}...")

        try:
            for feature_name, feature_value in extractor.extract_features(video_path):
                result_queue.put({
                    "video_path": video_path,
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                })
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

            logger.info(f"[DB Write Thread {writer_thread_id}] Writing {data['video_path']}, {data['feature_name']} to DB...")
            db.insert(data['feature_name'], data['video_path'], data['feature_value'])
            logger.info(f"[DB Write Thread {writer_thread_id}] Successfully wrote {data['video_path']}, {data['feature_name']} to DB.")
            result_queue.task_done()
        except Exception as e:
            logger.error(f"[DB Write Thread {writer_thread_id}] Error while inserting into DB: {e}")
            result_queue.task_done()