import torch
import time
import requests
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image
from PIL import Image
from io import BytesIO
import statistics

model_name = "../../models/VL3-SigLIP-NaViT"
image_url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
device = "cuda:0"
NUM_RUNS = 200  # 循环200次

class PipelineTimer:
    def __init__(self):
        self.times = {
            'download': [],
            'preprocess': [],
            'to_gpu': [],
            'forward': [],
            'to_cpu': []
        }
        
    def add_time(self, phase, duration):
        self.times[phase].append(duration)
    
    def print_stats(self):
        print("\n" + "="*60)
        print("📊 Pipeline Performance Statistics")
        print("="*60)
        
        for phase, times in self.times.items():
            if times:
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                min_time = min(times)
                max_time = max(times)
                
                print(f"{phase.capitalize():>12}: "
                      f"avg={avg_time:.3f}s ± {std_time:.3f}s, "
                      f"min={min_time:.3f}s, max={max_time:.3f}s")
                
                # 标记异常值（超过平均值2倍标准差）
                outliers = [t for t in times if abs(t - avg_time) > 2 * std_time]
                if outliers:
                    print(f"{'':>12}  ⚠️  {len(outliers)} outliers detected: {outliers[:5]}")
        
        total_avg = sum(statistics.mean(times) for times in self.times.values())
        print(f"\n{'Total avg':>12}: {total_avg:.3f}s per frame")

def load_model():
    print("🔁 Loading model with FlashAttention2...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model

def download_image_from_url(url):
    """模拟从URL获取图片"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def run_pipeline_test():
    print("📥 Initializing...")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = load_model()
    
    # 预热GPU
    print("🔥 Warming up GPU...")
    dummy_image = load_image(image_url)
    for _ in range(3):
        inputs = processor(images=dummy_image, merge_size=1)
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()
    print("✅ Warmup completed")
    
    timer = PipelineTimer()
    
    print(f"\n🚀 Starting pipeline test ({NUM_RUNS} iterations)...")
    
    for i in range(NUM_RUNS):
        # Phase 1: Download image from URL
        t0 = time.time()
        image = download_image_from_url(image_url)
        t1 = time.time()
        timer.add_time('download', t1 - t0)
        
        # Phase 2: Preprocess
        t2 = time.time()
        inputs = processor(images=image, merge_size=1)
        t3 = time.time()
        timer.add_time('preprocess', t3 - t2)
        
        # Phase 3: Move to GPU
        t4 = time.time()
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        t5 = time.time()
        timer.add_time('to_gpu', t5 - t4)
        
        # Phase 4: Forward pass
        t6 = time.time()
        with torch.no_grad():
            embeddings = model(**inputs)
        torch.cuda.synchronize()  # 确保计时准确
        t7 = time.time()
        timer.add_time('forward', t7 - t6)
        
        # Phase 5: Move to CPU
        t8 = time.time()
        embeddings_cpu = embeddings.detach().cpu()
        del embeddings
        t9 = time.time()
        timer.add_time('to_cpu', t9 - t8)
        
        # 打印当前帧的统计信息
        print(f"Frame {i:3d}: download={t1-t0:.3f}s, preprocess={t3-t2:.3f}s, "
              f"to_gpu={t5-t4:.3f}s, forward={t7-t6:.3f}s, to_cpu={t9-t8:.3f}s, "
              f"total={t9-t0:.3f}s")
        
        # 清理内存
        del inputs, embeddings_cpu
    
    print(f"✅ Completed {NUM_RUNS} iterations")
    timer.print_stats()
    
    # 分析异常情况
    print("\n🔍 Analyzing performance anomalies...")
    for phase, times in timer.times.items():
        if times:
            avg = statistics.mean(times)
            slow_frames = [(i, t) for i, t in enumerate(times) if t > avg * 3]
            if slow_frames:
                print(f"{phase}: {len(slow_frames)} frames were >3x slower than average")
                print(f"  Slow frame indices: {[i for i, t in slow_frames[:10]]}")

if __name__ == "__main__":
    try:
        run_pipeline_test()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()