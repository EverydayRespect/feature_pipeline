import torch
import time
import requests
import numpy as np
import gc
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from io import BytesIO
from typing import Optional, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

model_name = "../../models/VL3-SigLIP-NaViT"
image_url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
device = "cuda:0"
NUM_RUNS = 200

def load_video_mock(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = 1,
    max_frames: Optional[float] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    precise_time: bool = False,
    verbose: bool = False,
    temporal_factor: int = 1,
) -> Tuple[List, List[float]]:
    """
    Mock版本的load_video函数，返回PIL Image格式
    """
    num_frames = NUM_RUNS
    if max_frames is not None:
        num_frames = int(min(num_frames, max_frames))
    
    duration = num_frames / (fps if fps else 1.0)
    start_time = start_time if start_time is not None else 0.0
    
    # 加载基础图像
    if verbose:
        print(f"Mock loading video: {video_path}")
        print(f"Downloading base image...")
    
    response = requests.get(image_url)
    base_image = Image.open(BytesIO(response.content))
    
    # 处理size参数
    if size is not None:
        w, h = base_image.size
        scale_factor = size / min(w, h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        new_w = new_w // size_divisible * size_divisible
        new_h = new_h // size_divisible * size_divisible
        base_image = base_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if verbose:
            print(f"Resized image to {new_w}x{new_h}")
    
    # 生成frames
    frames = []
    timestamps = []
    
    if verbose:
        print(f"Generating {num_frames} PIL Image frames...")
    
    for i in range(num_frames):
        if i == 0:
            frame = base_image.copy()
        else:
            # 添加微小变化
            img_array = np.array(base_image)
            noise = np.random.randint(-1, 2, img_array.shape, dtype=np.int16)
            noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frame = Image.fromarray(noisy_array)
        
        frames.append(frame)
        timestamp = start_time + (i * duration / num_frames)
        timestamps.append(timestamp)
    
    if verbose:
        print(f"Generated {len(frames)} PIL Image frames, size: {frames[0].size}")
    
    return frames, timestamps


class VideoProcessor:
    """完全模拟你的VideoProcessor类"""
    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.gpu_thread_id = 0
        
        print(f"🔁 Loading model on {device}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
    
    def extract_embeddings(self, frames):
        """完全复制你的extract_embeddings方法逻辑"""
        
        # 在开始前强制清理（测试内存管理影响）
        gc.collect()
        torch.cuda.empty_cache()
        
        for frame_id, frame in enumerate(frames):
            # 每隔50帧强制同步和清理（测试是否能减少问题）
            if frame_id % 50 == 0:
                torch.cuda.synchronize(self.device)
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Cleaned cache at frame {frame_id}")
            
            t0 = time.time()
            # -------- Phase 1: Preprocessing --------
            image_input = self.processor(
                frame, merge_size=1, return_tensors="pt",
            )
            t1 = time.time()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Frame {frame_id}: preprocessing {t1 - t0:.3f} sec")

            # -------- Phase 2: Grid sizes --------
            if "grid_sizes" in image_input:
                *_, grid_row, grid_col = image_input["grid_sizes"][0]
            else:
                raise ValueError("grid_sizes not found in image_input")

            # -------- Phase 3: Move to GPU --------
            t2 = time.time()
            image_input = {k: v.cuda(device=self.device) for k, v in image_input.items()}
            if "pixel_values" in image_input:
                image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
            else:
                raise ValueError("pixel_values not found in image_input")
            t3 = time.time()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Frame {frame_id}: move_to_gpu {t3 - t2:.3f} sec")

            # -------- Phase 4: Forward pass --------
            t4 = time.time()
            with torch.no_grad():
                embeddings = self.model(**image_input)
            torch.cuda.synchronize(self.device)  # 确保计时准确
            t5 = time.time()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Frame {frame_id}: forward {t5 - t4:.3f} sec")

            # -------- Phase 5: Move to CPU --------
            embeddings_cpu = embeddings.detach().cpu()
            del embeddings
            t6 = time.time()
            logger.info(f"[GPU-{self.device}-Thread-{self.gpu_thread_id}] Frame {frame_id}: move_to_cpu {t6 - t5:.3f} sec")

            yield frame_id, grid_row, grid_col, embeddings_cpu


def run_complete_pipeline_test():
    """完全模拟你的实际pipeline调用方式"""
    
    print("📥 Initializing complete pipeline test...")
    
    # 1. 加载视频数据（使用mock函数）
    print("🎬 Loading video data...")
    video_data, timestamps = load_video_mock(
        "mock_video.mp4", 
        fps=1, 
        max_frames=NUM_RUNS, 
        verbose=True
    )
    
    # 2. 创建processor（模拟你的类实例）
    processor = VideoProcessor(device=device)
    
    # 3. 预热GPU
    print("🔥 Warming up GPU...")
    warmup_frames = video_data[:3]
    for _ in processor.extract_embeddings(warmup_frames):
        pass  # 只是预热，不处理结果
    print("✅ GPU warmup completed")
    
    # 4. 运行完整的pipeline（完全模拟你的实际调用）
    print(f"\n🚀 Starting complete pipeline test ({NUM_RUNS} frames)...")
    
    total_start_time = time.time()
    processed_count = 0
    
    # 这里完全模拟你的实际调用方式
    for frame_id, grid_rows, grid_cols, embeddings in processor.extract_embeddings(video_data):
        # 模拟你对结果的处理
        result = {
            "frame_id": frame_id,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "embeddings": embeddings.tolist()  # 这行可能很耗时
        }
        
        processed_count += 1
        
        # 定期报告进度
        if processed_count % 50 == 0:
            elapsed = time.time() - total_start_time
            avg_time = elapsed / processed_count
            print(f"Progress: {processed_count}/{NUM_RUNS}, avg time per frame: {avg_time:.3f}s")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_time_per_frame = total_time / NUM_RUNS
    
    print(f"\n✅ Pipeline test completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.3f} seconds")
    print(f"Processed {processed_count} frames")


def run_direct_comparison():
    """运行直接对比：生成器方式 vs 批处理方式"""
    
    print("\n" + "="*60)
    print("🔍 Running direct comparison: Generator vs Batch processing")
    print("="*60)
    
    # 准备数据
    video_data, _ = load_video_mock("mock_video.mp4", fps=1, max_frames=50, verbose=False)
    processor = VideoProcessor(device=device)
    
    # 方式1：生成器方式（你当前的方式）
    print("\n1️⃣ Testing Generator approach (your current way):")
    start_time = time.time()
    count = 0
    for frame_id, grid_rows, grid_cols, embeddings in processor.extract_embeddings(video_data):
        result = {
            "frame_id": frame_id,
            "embeddings": embeddings.tolist()
        }
        count += 1
    generator_time = time.time() - start_time
    print(f"Generator approach: {generator_time:.3f}s for {count} frames")
    
    # 清理
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    
    # 方式2：批处理方式（测试脚本的方式）
    print("\n2️⃣ Testing Batch approach (test script way):")
    start_time = time.time()
    results = []
    
    for frame_id, frame in enumerate(video_data):
        # 直接调用处理逻辑，不使用生成器
        t0 = time.time()
        image_input = processor.processor(frame, merge_size=1, return_tensors="pt")
        image_input = {k: v.cuda(device=processor.device) for k, v in image_input.items()}
        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].to(torch.bfloat16)
        
        with torch.no_grad():
            embeddings = processor.model(**image_input)
        torch.cuda.synchronize(processor.device)
        
        embeddings_cpu = embeddings.detach().cpu()
        del embeddings
        
        result = {
            "frame_id": frame_id,
            "embeddings": embeddings_cpu.tolist()
        }
        results.append(result)
        
        t1 = time.time()
        if frame_id % 10 == 0:
            print(f"  Batch frame {frame_id}: {t1-t0:.3f}s")
    
    batch_time = time.time() - start_time
    print(f"Batch approach: {batch_time:.3f}s for {len(results)} frames")
    
    print(f"\n📊 Comparison Result:")
    print(f"Generator time: {generator_time:.3f}s")
    print(f"Batch time: {batch_time:.3f}s")
    print(f"Difference: {abs(generator_time - batch_time):.3f}s")
    
    if generator_time > batch_time * 1.1:
        print("⚠️  Generator approach is significantly slower - this might be the issue!")
    else:
        print("✅ Both approaches have similar performance")


if __name__ == "__main__":
    try:
        # 运行完整的pipeline测试
        run_complete_pipeline_test()
        
        # 运行对比测试
        run_direct_comparison()
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()