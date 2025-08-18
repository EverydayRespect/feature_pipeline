import re
import time
import torch
from logger import logger
from transformers import AutoModel, AutoImageProcessor

from models.base import BaseModel


class VL3SigLIPExtractor(BaseModel):
    
    def __init__(self, model_name, model_path, feature_list, device, gpu_thread_id):
        super().__init__(model_name, model_path, feature_list, device, gpu_thread_id)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="flash_attention_2"
        ).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    @torch.inference_mode()
    def extract_embeddings(self, frames):
        
        for frame_id, frame in enumerate(frames):
            image_input = self.processor(
                frame,
                merge_size=1,
                return_tensors="pt",
            )

            if "grid_sizes" in image_input:
                _, grid_row, grid_col = image_input["grid_sizes"][0]
            else:
                raise ValueError("grid_sizes not found in image_input")

            image_input = {k: v.cuda(device=self.device) for k, v in image_input.items()}
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")
            
            embeddings = self.model(**image_input)
            # Move embeddings to CPU and detach from graph
            embeddings_cpu = embeddings.detach().cpu()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] extracted one frame embedding...")
            # Free GPU memory
            del embeddings

            yield frame_id, grid_row, grid_col, embeddings_cpu
    
    @torch.inference_mode()
    def extract_embeddings_pooling(self, frames):
        
        all_embeddings = []
        for _, frame in enumerate(frames):
            image_input = self.processor(
                frame,
                merge_size=1,
                return_tensors="pt",
            )

            image_input = {k: v.cuda(device=self.device) for k, v in image_input.items()}
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")
            
            embeddings = self.model(**image_input)
            # Move embeddings to CPU and detach from graph
            frame_embedding = embeddings.mean(dim=0).detach().cpu()
            # Free GPU memory
            all_embeddings.append(frame_embedding)
            del embeddings
            
        return all_embeddings