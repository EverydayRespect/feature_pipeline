from huggingface_hub import snapshot_download
local_dir = "./VL3-SigLIP-NaViT"

snapshot_download(
    repo_id="DAMO-NLP-SG/VL3-SigLIP-NaViT",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)