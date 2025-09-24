import torch
import time
import threading
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

model_name = "../models/VL3-SigLIP-NaViT"
image_url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
device = "cuda:0"
NUM_RUNS = 20

# 🔒 全局锁：保证模型加载不会并发
load_lock = threading.Lock()

def load_model(use_flash_attn):
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    print(f"\n🔁 Loading model with attn_implementation = {attn_impl}...")
    with load_lock:  # 加锁避免 meta tensor 并发冲突
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map={"": device},   # 这里可以继续用 device_map
        )
    return model

def run_inference(model, processor, image_tensor, num_runs=NUM_RUNS):
    inputs = processor(images=image_tensor, merge_size=1)
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    return sum(times) / len(times)

def worker(thread_id, results):
    try:
        print(f"🧵 Thread {thread_id} starting...")
        image_tensor = load_image(image_url)
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        model = load_model(use_flash_attn=True)
        avg_time = run_inference(model, processor, image_tensor)

        results[thread_id] = avg_time
        print(f"🧵 Thread {thread_id} finished. Avg time: {avg_time:.2f} sec")

    except Exception as e:
        print(f"❌ Thread {thread_id} failed: {e}")
        results[thread_id] = None

def main():
    threads = []
    results = {}

    for i in range(2):
        t = threading.Thread(target=worker, args=(i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== 多线程结果 ===")
    for i in range(2):
        if results[i] is not None:
            print(f"Thread {i} Avg Inference Time: {results[i]:.2f} sec")
        else:
            print(f"Thread {i} failed.")

if __name__ == "__main__":
    main()
