import threading
from logger import logger
from models.base import BaseModel
from models.clip import CLIPExtractor
from models.vl3siglip import VL3SigLIPExtractor
from models.beats import BEATsExtractor
from models.wavlm import WavLMExtractor
from models.phi4_audio import Phi4MelExtractor
# Global lock for thread-safe model loading
model_load_lock = threading.Lock()

model_map = {
    "CLIP": CLIPExtractor,
    "VL3-SigLIP-NaViT": VL3SigLIPExtractor,
    "BEATs": BEATsExtractor,
    "Phi-4-mel": Phi4MelExtractor,
    "WavLM-Large": WavLMExtractor,
}

def load_model(gpu_id, gpu_thread_id, model_conf) -> BaseModel:
    model_name = model_conf["name"]
    model_path = model_conf["path"]
    feature_list = model_conf["features"]
    logger.info(f"Loading {model_name} from {model_path} onto device {gpu_id}...")

    if model_name not in model_map:
        raise ValueError(f"Model {model_name} is not supported.")

    model_class = model_map[model_name]

    # 🔒 Thread-safe instantiation
    with model_load_lock:
        return model_class(model_name, model_path, feature_list, gpu_id, gpu_thread_id)
