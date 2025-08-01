import re
import time
import torch
from logger import logger
from transformers import AutoModel, AutoImageProcessor

from models.base import BaseModel


class VL3SigLIPExtractor(BaseModel):
    
    def __init__(self, model_name, model_path, feature_list, device_id):
        super().__init__(model_name, model_path, feature_list, device_id)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
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

            if "grid_sizes" in image_input:
                _, grid_row, grid_col = image_input["grid_sizes"][0]
            else:
                raise ValueError("grid_sizes not found in image_input")

            image_input = {k: torch.tensor(v).cuda() for k, v in image_input.items()}
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")
            
            embeddings = self.model(**image_input)
            # Move embeddings to CPU and detach from graph
            embeddings_cpu = embeddings.detach().cpu()
            # Free GPU memory
            del embeddings

            logger.info(f"[GPU-{self.device}] Extracted {embeddings_cpu.shape} embeddings from frame {frame_id}.")
            yield frame_id, grid_row, grid_col, embeddings_cpu
 