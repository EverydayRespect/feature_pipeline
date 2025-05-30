from logger import logger
from models.base import BaseModel
from models.clip import CLIPExtractor
from models.vl3siglip import VL3SigLIPExtractor


model_map = {
    "CLIP": CLIPExtractor,
    "VL3-SigLIP-NaViT": VL3SigLIPExtractor,
}

def load_model(gpu_id, model_conf) -> BaseModel:
    model_name = model_conf["name"]
    model_path = model_conf["path"]
    feature_list = model_conf["features"]
    logger.info(f"Loading {model_name} from {model_path} onto device {gpu_id}...")
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model_class = model_map[model_name]
    return model_class(model_name, model_path, feature_list, gpu_id)