from abc import abstractmethod
from logger import logger
from utils import load_video

class BaseModel:

    def __init__(self, model_name, model_path, feature_list, device=None, gpu_thread_id=0):
        self.model_name = model_name
        self.model_path = model_path
        self.feature_list = feature_list
        self.device = device or 'cpu'
        self.gpu_thread_id = gpu_thread_id

        self.feature_func_map = {
            "video_embedding": self.extract_embeddings,
            "video_embedding_pooling": self.extract_embeddings_pooling
        }

    def load_video(self, video_path):
        # extract video frames and process
        logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Loading video {video_path}...")
        frames, _ = load_video(video_path)
        logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Loaded {len(frames)} frames from {video_path}.")
        
        return frames
    
    @abstractmethod
    def extract_embeddings(self, data):
        raise NotImplementedError("extract_embeddings should be overridden by subclasses")
    
    @abstractmethod
    def extract_embeddings_pooling(self, data):
        raise NotImplementedError("extract_embeddings_pooling should be overridden by subclasses")

    def extract_features(self, video_path):
        video_data = self.load_video(video_path)
        if "video_embedding" in self.feature_list:
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Extracting video embeddings from {video_path}...")
            for frame_id, grid_rows, grid_cols, embeddings in self.extract_embeddings(video_data):
                yield "video_embedding", {
                    "frame_id": frame_id,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "embeddings": embeddings.tolist()
                }
        if "video_embedding_pooling" in self.feature_list:
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Extracting video pooling embeddings from {video_path}...")
            frame_embeddings = self.extract_embeddings_pooling(video_data)
            yield "video_embedding_pooling", {
                "frame_embeddings": frame_embeddings
            }