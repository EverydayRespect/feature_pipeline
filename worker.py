from database import BaseVectorDB, init_db
from logger import logger
from models.utils import load_model

def gpu_worker_thread(gpu_id, task_queue, result_queue, model_conf):
    
    try:
        extractor = load_model(gpu_id, model_conf)
    except Exception as e:
        logger.error(f"[GPU-{gpu_id}] Failed to load model: {e}")
        logger.info(f"[GPU-{gpu_id}] Worker exiting due to model load failure.")
        return 
    
    while True:
        try:
            video_path = task_queue.get(timeout=2)
        except:
            logger.warning(f"[GPU-{gpu_id}] No more tasks. Exiting.")
            break

        logger.info(f"[GPU-{gpu_id}] Start processing {video_path}...")

        try:
            for feature_name, feature_value in extractor.extract_features(video_path):
                result_queue.put({
                    "video_path": video_path,
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                })
        except Exception as e:
            logger.error(f"[GPU-{gpu_id}] Error processing {video_path}: {e}")
        finally:
            logger.info(f"[GPU-{gpu_id}] Finished processing {video_path}...")
            task_queue.task_done()

    logger.info(f"[GPU-{gpu_id}] Worker exiting.")

def db_write_thread(result_queue, db: BaseVectorDB):
    while True:
        try:
            data = result_queue.get()
            if data is None:
                # Signal to exit
                result_queue.task_done()
                break

            logger.info(f"[DB Write Thread] Writing {data['video_path']}, {data['feature_name']} to DB...")
            db.insert(data['feature_name'], data['video_path'], data['feature_value'])
            logger.info(f"[DB Write Thread] Successfully wrote {data['video_path']}, {data['feature_name']} to DB.")
            result_queue.task_done()
        except Exception as e:
            logger.error(f"[DB Write Thread] Error while inserting into DB: {e}")
            result_queue.task_done()