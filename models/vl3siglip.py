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
    def extract_embeddings(self, frames, batch_size=60):
        """
        Encode frames in batches (e.g., 60 frames per forward pass).
        Each batch is preprocessed, encoded, and yielded per frame.
        """
        num_frames = len(frames)
        logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Total frames: {num_frames}, batch_size={batch_size}")

        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch_frames = frames[start:end]

            t0 = time.time()
            # -------- Phase 1: Preprocessing (batch of frames) --------
            image_input = self.processor(
                batch_frames,
                merge_size=1,
                return_tensors="pt",
            )
            t1 = time.time()
            logger.info(
                f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] "
                f"Frames {start}-{end-1}: preprocessing {t1 - t0:.3f} sec"
            )

            if "grid_sizes" not in image_input:
                raise ValueError("grid_sizes not found in image_input")
            grid_sizes = image_input["grid_sizes"]  # (batch_size, 3)

            # Move inputs to GPU
            image_input = {k: v.cuda(device=self.device) for k, v in image_input.items()}
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")

            # -------- Phase 2: Forward pass (encode the batch) --------
            t2 = time.time()
            embeddings = self.model(**image_input)
            torch.cuda.synchronize(self.device)
            t3 = time.time()
            logger.info(
                f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] "
                f"Frames {start}-{end-1}: forward {t3 - t2:.3f} sec"
            )

            # -------- Phase 3: Move to CPU & split per frame --------
            embeddings_cpu = embeddings.detach().cpu()
            del embeddings
            yield embeddings_cpu

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