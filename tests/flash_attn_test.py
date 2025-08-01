import torch
from flash_attn import flash_attn_func

print("CUDA available:", torch.cuda.is_available())

q = torch.randn(2, 8, 32, 64, device="cuda").half()  # convert to fp16
k = torch.randn(2, 8, 32, 64, device="cuda").half()
v = torch.randn(2, 8, 32, 64, device="cuda").half()

out = flash_attn_func(q, k, v, causal=False)
print("Output shape:", out.shape)
