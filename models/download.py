import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
from huggingface_hub import snapshot_download
# local_dir = "./VL3-SigLIP-NaViT"

# snapshot_download(
#     repo_id="DAMO-NLP-SG/VL3-SigLIP-NaViT",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False
# )

# local_dir = "./SigLIP-Base"

# snapshot_download(
#     repo_id="google/siglip-base-patch16-224",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False
# )

# wavlm_dir = "models/WavLM-Large" 
# snapshot_download(
#     repo_id="microsoft/wavlm-large",
#     local_dir=wavlm_dir,
#     local_dir_use_symlinks=False
# )

qwen3_asr_dir = "models/Qwen3-ASR-0.6B"
snapshot_download(
    repo_id="Qwen/Qwen3-ASR-0.6B",
    local_dir=qwen3_asr_dir,
    local_dir_use_symlinks=False,  # 可省略
)
clip14_dir = "models/CLIP-ViT-L-14"
clip14_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
def download_clip14(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

download_clip14(clip14_url, clip14_dir)
