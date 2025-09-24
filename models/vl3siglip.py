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
            t0 = time.time()
            # -------- Phase 1: Preprocessing --------
            image_input = self.processor(
                frame,
                merge_size=1,
                return_tensors="pt",
            )
            t1 = time.time()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Frame {frame_id}: preprocessing {t1 - t0:.3f} sec")

            if "grid_sizes" in image_input:
                _, grid_row, grid_col = image_input["grid_sizes"][0]
            else:
                raise ValueError("grid_sizes not found in image_input")
            image_input = {k: v.cuda(device=self.device) for k, v in image_input.items()}
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")

            # -------- Phase 4: Forward pass --------
            t4 = time.time()
            embeddings = self.model(**image_input)
            torch.cuda.synchronize(self.device)  # 确保计时准确
            t5 = time.time()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Frame {frame_id}: forward {t5 - t4:.3f} sec")

            # -------- Phase 5: Move to CPU --------
            embeddings_cpu = embeddings.detach().cpu()
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