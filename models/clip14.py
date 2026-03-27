# from logger import logger
import torch
import clip
from PIL import Image
from models.base import BaseModel
import re
import time
import torch
from logger import logger
from models.base import BaseModel


class CLIP14Extractor(BaseModel):
    
    def __init__(self, model_name, model_path, feature_list, device, gpu_thread_id):
        super().__init__(model_name, model_path, feature_list, device, gpu_thread_id)
        self.model, self.processor = clip.load(model_path, device=device)

    @torch.inference_mode()
    def extract_embeddings(self, frames, batch_size=60):
        """
        Encode frames using CLIP in batches.
        Each batch is preprocessed, encoded, and yielded.
        """

        num_frames = len(frames)
        logger.info(
            f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] "
            f"Total frames: {num_frames}, batch_size={batch_size}"
        )

        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch_frames = frames[start:end]

            # -------- Phase 1: Preprocessing --------
            t0 = time.time()

            # CLIP preprocess returns tensor [3, H, W]
            image_tensors = torch.stack(
                [self.processor(img) for img in batch_frames]
            )

            t1 = time.time()
            logger.info(
                f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] "
                f"Frames {start}-{end-1}: preprocessing {t1 - t0:.3f} sec"
            )

            # Move to GPU
            image_tensors = image_tensors.to(self.device)

            # -------- Phase 2: Forward pass --------
            t2 = time.time()

            embeddings = self.model.encode_image(image_tensors)

            torch.cuda.synchronize(self.device)

            t3 = time.time()
            logger.info(
                f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] "
                f"Frames {start}-{end-1}: forward {t3 - t2:.3f} sec"
            )

            # -------- Phase 3: Normalize + Move to CPU --------
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            embeddings_cpu = embeddings.detach().cpu()

            del embeddings
            yield embeddings_cpu
