import torch
import time
import threading
from transformers import SiglipVisionModel, AutoImageProcessor
from PIL import Image
import requests

# 模型路径（可以用本地路径，或者直接用 Hugging Face hub 的 "google/siglip-base-patch16-224"）
model_name = "../../models/SigLIP-Base"
device = "cuda:0"
NUM_RUNS = 20
WARMUP_RUNS = 5
BATCH_SIZE = 128

# 🔒 全局锁：保证模型加载不会并发
load_lock = threading.Lock()

def load_model():
    print(f"\n🔁 Loading model {model_name} ...")
    with load_lock:
        model = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(device)
    return model

def run_inference(model, processor, image, batch_size=BATCH_SIZE, num_runs=NUM_RUNS, warmup=WARMUP_RUNS):
    images = [image] * batch_size
    inputs = processor(images=images, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # 🚀 warmup
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

    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time
    return avg_time, throughput

def worker(thread_id, results, image):
    try:
        print(f"🧵 Thread {thread_id} starting...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = load_model()

        avg_time, throughput = run_inference(model, processor, image)
        results[thread_id] = (avg_time, throughput)

        print(f"🧵 Thread {thread_id} finished. "
              f"Avg time: {avg_time:.4f} sec | Throughput: {throughput:.1f} images/sec")

    except Exception as e:
        print(f"❌ Thread {thread_id} failed: {e}")
        results[thread_id] = None

def main():
    # 测试图片
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    threads = []
    results = {}

    for i in range(2):
        t = threading.Thread(target=worker, args=(i, results, image))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== 多线程结果 (batch=128) ===")
    total_throughput = 0.0
    for i in range(2):
        if results[i] is not None:
            avg_time, throughput = results[i]
            total_throughput += throughput
            print(f"Thread {i} → Avg time: {avg_time:.4f} sec | "
                  f"Throughput: {throughput:.1f} images/sec")
        else:
            print(f"Thread {i} failed.")

    print(f"\n🚀 总吞吐量 (2线程): {total_throughput:.1f} images/sec")

if __name__ == "__main__":
    main()
