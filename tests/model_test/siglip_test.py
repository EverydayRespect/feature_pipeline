import torch
import time
from transformers import AutoImageProcessor, SiglipVisionModel
from PIL import Image
import requests

# Model path (change to your local SigLIP-Base)
model_name = "../../models/SigLIP-Base"
device = "cuda:0"
NUM_RUNS = 20   # runs per batch size for averaging
WARMUP_RUNS = 5 # warmup runs
batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]

def load_model():
    print(f"\n🔁 Loading model {model_name} ...")
    model = SiglipVisionModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)
    return model

def run_inference(model, processor, image, batch_size, num_runs=NUM_RUNS, warmup=WARMUP_RUNS):
    # Preprocessing timing (run once for fair measurement)
    t0 = time.time()
    images = [image] * batch_size
    inputs = processor(images=images, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    preprocess_time = time.time() - t0

    print(f"📐 Preprocessed pixel_values shape: {inputs['pixel_values'].shape}")
    # Move to GPU (exclude from preprocess time)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 🚀 Warmup (not counted)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)

    # Measure forward time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    avg_forward = sum(times) / len(times)
    return preprocess_time, avg_forward

def main():
    # Download a test image
    url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = load_model()

    print("\n=== Batch size Benchmark (with preprocess) ===")
    for bs in batch_size_list:
        preprocess_time, avg_time = run_inference(model, processor, image, bs)
        throughput = bs / avg_time
        print(
            f"Batch size {bs:<3d} → "
            f"Preprocess: {preprocess_time:.4f} sec | "
            f"Forward: {avg_time:.4f} sec | "
            f"Throughput: {throughput:.1f} images/sec"
        )

if __name__ == "__main__":
    main()
