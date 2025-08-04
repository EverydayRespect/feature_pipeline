import threading
import torch
from transformers import AutoModel, AutoImageProcessor

MODEL_PATH = "../models/VL3-SigLIP-NaViT"
NUM_GPUS = 3
THREADS_PER_GPU = 8

# Global lock for safe model loading
model_load_lock = threading.Lock()

class ModelLoaderThread(threading.Thread):
    def __init__(self, gpu_id, thread_id, results):
        super().__init__()
        self.gpu_id = gpu_id
        self.thread_id = thread_id
        self.results = results

    def run(self):
        try:
            device = f"cuda:{self.gpu_id}"
            print(f"[Thread {self.thread_id}] Loading model on {device}...")

            with model_load_lock:
                model = AutoModel.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
                model = model.to(device)

                processor = AutoImageProcessor.from_pretrained(
                    MODEL_PATH, trust_remote_code=True
                )

            _ = next(model.parameters()).device  # Confirm loaded
            print(f"[Thread {self.thread_id}] Model loaded successfully on {device}")
            self.results[self.thread_id] = True
        except Exception as e:
            print(f"[Thread {self.thread_id}] Failed to load model on {device}: {e}")
            self.results[self.thread_id] = False

def main():
    threads = []
    results = {}
    thread_id = 0

    for gpu_id in range(NUM_GPUS):
        for _ in range(THREADS_PER_GPU):
            t = ModelLoaderThread(gpu_id=gpu_id, thread_id=thread_id, results=results)
            threads.append(t)
            thread_id += 1

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    successes = sum(results.values())
    print(f"\n✅ {successes} out of {len(threads)} models loaded successfully.")

if __name__ == "__main__":
    main()
