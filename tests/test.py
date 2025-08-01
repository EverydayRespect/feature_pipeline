import torch
import time
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image


def repeat_input_dict_batch_dim(input_dict, batch_size):
    repeated = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            # If first dim is tokens, add batch dim in front and repeat along batch
            if v.dim() == 2:
                # v shape: [tokens, feature_dim]
                v = v.unsqueeze(0)  # [1, tokens, feature_dim]
                repeated[k] = v.repeat(batch_size, 1, 1).cuda()
            elif v.dim() == 3:
                # Already has batch dim, repeat along batch dim
                repeated[k] = v.repeat(batch_size, 1, 1).cuda()
            else:
                # For other dims, just move to cuda
                repeated[k] = v.cuda()
        elif isinstance(v, list):
            repeated[k] = v * batch_size
        else:
            repeated[k] = v
    return repeated


def test_batch_inference(model_path: str, image_url: str, batch_sizes=[1, 2, 4, 8]):
    print(f"Loading image from: {image_url}")
    single_image = load_image(image_url)
    print("Single image size:", single_image.size)

    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Process the single image once
    original_inputs = processor(images=single_image, merge_size=1, return_tensors="pt")

    print("\n--- Batch Inference Timing and Memory Usage ---")
    for batch_size in batch_sizes:
        inputs = repeat_input_dict_batch_dim(original_inputs, batch_size)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        print(f"\nBatch size {batch_size:<2} | pixel_values shape: {inputs['pixel_values'].shape}")

        # Warm-up
        _ = model(**inputs)

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()

        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        print(f" -> Inference time: {end - start:.4f} sec")
        print(f" -> GPU memory allocated before: {mem_before / (1024**3):.2f} GiB")
        print(f" -> GPU memory allocated after:  {mem_after / (1024**3):.2f} GiB")
        print(f" -> Peak GPU memory usage:       {peak_mem / (1024**3):.2f} GiB")

        # Free memory
        del outputs
        del inputs
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    model_path = "../models/VL3-SigLIP-NaViT"
    image_url = "https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/assets/sora.png?raw=true"
    test_batch_inference(model_path, image_url)
