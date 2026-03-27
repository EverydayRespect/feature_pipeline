import torch
import clip
from PIL import Image
import requests
from io import BytesIO

# -------------------------
# 1️⃣ Load model
# -------------------------
device = "mps"

model, preprocess = clip.load(
    name="/Users/johan/Desktop/feature_pipeline_lab/models/models/CLIP-ViT-L-14/ViT-L-14.pt",
    device=device
)

model.eval()

# -------------------------
# 2️⃣ Load image from URL
# -------------------------
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

print("Original image resolution:", image.size)  
# (width, height)

# -------------------------
# 3️⃣ Preprocess single image
# -------------------------
image_tensor = preprocess(image)

print("After preprocess shape:", image_tensor.shape)
# [3, H, W]

# -------------------------
# 4️⃣ Stack into batch
# -------------------------
batch = torch.stack([image_tensor] * 4)  # simulate batch of 4

print("Batch shape:", batch.shape)
# [B, 3, H, W]

batch = batch.to(device)

# -------------------------
# 5️⃣ Encode
# -------------------------
with torch.inference_mode():
    embeddings = model.encode_image(batch)

print("Embedding shape:", embeddings.shape)
# [B, embed_dim]

# -------------------------
# 6️⃣ Print embedding dim
# -------------------------
print("Embedding dimension per image:", embeddings.shape[-1])
