import torch
import time
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

model_name = "../../models/VL3-SigLIP-NaViT"
image_url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
device = "cuda:0"
NUM_RUNS = 20  # 跑多少次取平均

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

def run_inference(model, processor, image_tensor, num_runs=NUM_RUNS):
    preprocess_times = []
    forward_times = []

    with torch.no_grad():
        for i in range(num_runs):
            # -------- Preprocess --------
            t0 = time.time()
            inputs = processor(images=image_tensor, merge_size=1)
            inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            t1 = time.time()
            preprocess_times.append(t1 - t0)

            # -------- Forward --------
            torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize()
            forward_times.append(time.time() - start)

    avg_preprocess = sum(preprocess_times) / len(preprocess_times)
    avg_forward = sum(forward_times) / len(forward_times)

    return avg_preprocess, avg_forward

def main():
    print("📥 Loading image...")
    image_tensor = load_image(image_url)
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Inference WITH FlashAttention
    model = load_model(use_flash_attn=True)
    avg_pre, avg_fwd = run_inference(model, processor, image_tensor)
    print(f"✅ Avg preprocess time over {NUM_RUNS} runs: {avg_pre:.3f} sec")
    print(f"✅ Avg forward time over {NUM_RUNS} runs: {avg_fwd:.3f} sec")

if __name__ == "__main__":
    main()
