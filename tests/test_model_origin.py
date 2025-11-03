import inspect
from transformers import AutoModel

model_path = "../models/VL3-SigLIP-NaViT"

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",   # 或 torch.bfloat16
    # attn_implementation="flash_attention_2",
)

# 打印模型类名
print("Model class:", model.__class__.__name__)

# 打印定义该类的文件路径
print("Defined in file:", inspect.getfile(model.__class__))

# 打印模块名
print("Module:", model.__class__.__module__)
