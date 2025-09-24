import torch
import time
from transformers import AutoImageProcessor, SiglipVisionModel
from PIL import Image
import requests

# 模型路径（改成本地存储的 SigLIP-Base）
model_name = "../../models/SigLIP-Base"
device = "cuda:0"
NUM_RUNS = 20   # 每个batch_size跑多少次取平均
WARMUP_RUNS = 5 # 预热次数
batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]

def load_model():
    print(f"\n🔁 Loading model {model_name} ...")
    model = SiglipVisionModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)
    return model

def run_inference(model, processor, image, batch_size, num_runs=NUM_RUNS, warmup=WARMUP_RUNS):
    # 构造 batch
    images = [image] * batch_size
    inputs = processor(images=images, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # 🚀 warmup，不计入统计
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)

    # 正式计时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    return sum(times) / len(times)

def main():
    # 下载一张测试图片
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = load_model()

    print("\n=== Batch size 测试结果 (已 warmup) ===")
    for bs in batch_size_list:
        avg_time = run_inference(model, processor, image, bs)
        throughput = bs / avg_time
        print(f"Batch size {bs:<2d} → Avg time: {avg_time:.4f} sec | Throughput: {throughput:.1f} images/sec")

if __name__ == "__main__":
    main()
