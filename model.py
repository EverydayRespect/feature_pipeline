from abc import abstractmethod
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoProcessor, CLIPVisionModel, AutoModel, AutoImageProcessor
from transformers.feature_extraction_utils import BatchFeature

from logger import logger
from utils import load_video

class BaseModel:

    def __init__(self, model_name, model_path, feature_list, device=None):
        self.model_name = model_name
        self.model_path = model_path
        self.feature_list = feature_list
        self.device = device or 'cpu'

        self.feature_func_map = {
            "video_embedding": self.extract_embeddings,
        }

    def load_video(self, video_path):
        # extract video frames and process
        logger.info(f"[GPU-{self.device}] Loading video {video_path}...")
        frames, _ = load_video(video_path)
        logger.info(f"[GPU-{self.device}] Loaded {len(frames)} frames from {video_path}.")
        
        return frames
    
    @abstractmethod
    def extract_embeddings(self, data):
        raise NotImplementedError("extract_embeddings should be overridden by subclasses")

    def extract_features(self, video_path):
        video_data = self.load_video(video_path)
        if "video_embedding" in self.feature_list:
            logger.info(f"[GPU-{self.device}] Extracting video embeddings from {video_path}...")
            for frame_id, grid_rows, grid_cols, embeddings in self.extract_embeddings(video_data):
                yield "video_embedding", {
                    "frame_id": frame_id,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "embeddings": embeddings.tolist()
                }
            

class VL3SigLIPExtractor(BaseModel):
    
    def __init__(self, model_name, model_path, feature_list, device_id):
        super().__init__(model_name, model_path, feature_list, device_id)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    @torch.inference_mode()
    def extract_embeddings(self, frames):
        
        for frame_id, frame in enumerate(frames):
            logger.info(f"[GPU-{self.device}] Processing frame {frame_id}...")
            image_input = self.processor(
                frame,
                merge_size=1,
                return_tensors="pt",
            )
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(self.device, dtype=torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")
            
            if "grid_sizes" in image_input:
                _, grid_row, grid_col = image_input["grid_sizes"][0]
            else:
                raise ValueError("grid_sizes not found in image_input")
            
            embeddings = self.model(**image_input)
            # Move embeddings to CPU and detach from graph
            embeddings_cpu = embeddings.detach().cpu()
            # Free GPU memory
            del embeddings

            logger.info(f"[GPU-{self.device}] Extracted {embeddings_cpu.shape} embeddings from frame {frame_id}.")
            yield frame_id, grid_row, grid_col, embeddings_cpu


class CLIPExtractor(BaseModel):

    def __init__(self, model_name, model_path, feature_list, device_id):
        super().__init__(model_name, model_path, feature_list, device_id)
        # Load the CLIP model here
        self.model = CLIPVisionModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    @torch.inference_mode()
    def extract_embeddings(self, frames, output="patch_embedding"):

        image_inputs = self.processor(
            images=frames,
            return_tensors="pt"
        )
        image_inputs = BatchFeature(data={**image_inputs})
        image_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}
        if "pixel_values" in image_inputs:
            image_inputs["pixel_values"] = image_inputs["pixel_values"].to(torch.bfloat16)

        model_outputs = self.model(**image_inputs)
        last_hidden_state = model_outputs.last_hidden_state
        frame_count, patch_count, hidden_dim = last_hidden_state.shape
        pooled_output = model_outputs.pooler_output

        if output == "pooler_output":
            return "pooler_output", {
                "frame_id": list(range(pooled_output.size(0))),
                "hidden_state": pooled_output.cpu().tolist()
            }
        elif output == "patch_embedding":
            flattened = last_hidden_state.reshape(-1, hidden_dim).cpu().tolist()
            return "video_embedding", {
                "frame_id": [i for i in range(frame_count) for _ in range(patch_count)],
                "patch_id": [j for _ in range(frame_count) for j in range(patch_count)],
                "embedding": flattened,
            }


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