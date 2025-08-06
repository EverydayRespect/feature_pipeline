import torch
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

model_name = "../models/VL3-SigLIP-NaViT"
image_path = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
images = load_image(image_path)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

inputs = processor(images=images, merge_size=1)
inputs = {k: torch.tensor(v).cuda() for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
image_features = model(**inputs)

print(image_features.shape)