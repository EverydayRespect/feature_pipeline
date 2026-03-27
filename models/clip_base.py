import torch
from transformers import AutoProcessor, CLIPVisionModel
from transformers.feature_extraction_utils import BatchFeature

from logger import logger
from models.base import BaseModel


class CLIPExtractor(BaseModel):

    def __init__(self, model_name, model_path, feature_list, device):
        super().__init__(model_name, model_path, feature_list, device)
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
                "embeddings": flattened,
            }

