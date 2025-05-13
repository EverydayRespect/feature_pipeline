import torch
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

model_name = "../models/VL3-SigLIP-NaViT"
image_path = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
images = load_image(image_path)
print(images.size)

# Test the model

# model = AutoModel.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     device_map="auto",
#     torch_dtype=torch.bfloat16
# )
# print(model)

# Test the image processor

processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

inputs = processor(images=[images], merge_size=1)

print(inputs['pixel_values'].shape)
# with torch.no_grad():
#     image_features = model(**inputs)
# print(image_features)
