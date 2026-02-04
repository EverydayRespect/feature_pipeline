import torch
import cv2
import time
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

model_path = "../models/VL3-SigLIP-NaViT"

# 1️⃣ Load model and processor
t0 = time.time()
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    attn_implementation="flash_attention_2"
)
processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
t1 = time.time()
print(f"[Load model+processor] {t1 - t0:.2f} sec")

# 2️⃣ Read frames from the video
def load_video_frames(video_path, max_frames=100, fps=1):
    """
    Extract frames at 1 frame per second (or your chosen fps).
    Returns a list of PIL images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, frame_rate // fps)
    print(f"[Video Info] {frame_count} frames, {frame_rate} fps, sampling interval={interval}")

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        i += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

video_path = "../video_samples/file_example_1.mp4"

t2 = time.time()
frames = load_video_frames(video_path, max_frames=60, fps=1)
t3 = time.time()
print(f"[Extract {len(frames)} frames] {t3 - t2:.2f} sec")

# 3️⃣ Preprocess frames
t4 = time.time()
inputs = processor(frames, merge_size=1, return_tensors="pt")
t5 = time.time()
print(f"[Preprocess frames] {t5 - t4:.2f} sec")
print("pixel_values:", inputs["pixel_values"].shape)

# 4️⃣ Prepare tensors
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
t6 = time.time()
print(f"[Move to GPU (if applicable)] {t6 - t5:.2f} sec")

# 5️⃣ Model inference
with torch.inference_mode():
    t7 = time.time()
    outputs = model(**inputs)
    t8 = time.time()
print(f"[Model forward] {t8 - t7:.2f} sec")

# 6️⃣ Inspect results
if isinstance(outputs, (list, tuple)):
    print("embeddings:", [o.shape for o in outputs])
elif isinstance(outputs, dict):
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"embeddings[{k}]:", v.shape)
else:
    print("embeddings:", outputs.shape)

print(f"[Total time] {t8 - t0:.2f} sec")

# Encode first 15 s and full 30 s separately
frames_1 = frames[:1]
t1_start = time.time()
inputs_1 = processor(frames_1, merge_size=1, return_tensors="pt")
if "pixel_values" in inputs_1:
    inputs_1["pixel_values"] = inputs_1["pixel_values"].to(torch.bfloat16)
with torch.inference_mode():
    out_1 = model(**inputs_1)
    t1_end = time.time()
    print(f"[Total time forward 1 frames] {t1_end - t1_start:.2f} sec")

frames_15 = frames[:15]
t15_start = time.time()
inputs_15 = processor(frames_15, merge_size=1, return_tensors="pt")
if "pixel_values" in inputs_15:
    inputs_15["pixel_values"] = inputs_15["pixel_values"].to(torch.bfloat16)
with torch.inference_mode():
    out_15 = model(**inputs_15)
    t15_end = time.time()
    print(f"[Total time forward 15 frames] {t15_end - t15_start:.2f} sec")

t30_start = time.time()
inputs_30 = processor(frames, merge_size=1, return_tensors="pt")
if "pixel_values" in inputs_30:
    inputs_30["pixel_values"] = inputs_30["pixel_values"].to(torch.bfloat16)
with torch.inference_mode():
    out_30 = model(**inputs_30)
    t30_end = time.time()
    print(f"[Total time forward 30 frames] {t30_end - t30_start:.2f} sec")

t60_start = time.time()
inputs_60 = processor(frames+frames, merge_size=1, return_tensors="pt")
if "pixel_values" in inputs_60:
    inputs_60["pixel_values"] = inputs_60["pixel_values"].to(torch.bfloat16)
with torch.inference_mode():
    out_60 = model(**inputs_60)
    t60_end = time.time()
    print(f"[Total time forward 60 frames] {t60_end - t60_start:.2f} sec")

# Compare the first patch of the first frame
diff = (out_15[0] - out_30[0]).abs().max()
print("Max |Δ| of first patch:", diff.item())

print(out_15[0][:10])
print(out_30[0][:10])

