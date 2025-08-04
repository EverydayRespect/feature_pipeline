import torch
import time
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

model_name = "../models/VL3-SigLIP-NaViT"
image_url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
device = "cuda:0"

def load_model(use_flash_attn):
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    print(f"\n🔁 Loading model with attn_implementation = {attn_impl}...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    return model

def run_inference(model, processor, image_tensor):
    inputs = processor(images=image_tensor, merge_size=1)
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        image_features = model(**inputs)
    torch.cuda.synchronize()
    duration = time.time() - start

    return image_features, duration

def main():
    print("📥 Loading image...")
    image_tensor = load_image(image_url)
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Inference WITHOUT FlashAttention
    model1 = load_model(use_flash_attn=False)
    _, time1 = run_inference(model1, processor, image_tensor)
    print(f"❌ Inference without FlashAttention took {time1:.2f} sec")
    del model1
    torch.cuda.empty_cache()

    # Inference WITH FlashAttention
    model2 = load_model(use_flash_attn=True)
    _, time2 = run_inference(model2, processor, image_tensor)
    print(f"✅ Inference with FlashAttention2 took {time2:.2f} sec")

if __name__ == "__main__":
    main()
