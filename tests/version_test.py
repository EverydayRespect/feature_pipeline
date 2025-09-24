import torch
import sys
import platform

def main():
    print("====== Environment Check for FlashAttention ======")

    # 1. Torch version
    torch_version = torch.__version__
    print(f"PyTorch version : {torch_version}")

    # 2. CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version    : {cuda_version}")

    # 3. CXX11 ABI
    try:
        cxx11abi = torch._C._GLIBCXX_USE_CXX11_ABI
    except AttributeError:
        cxx11abi = "Unknown"
    print(f"CXX11 ABI       : {cxx11abi}")

    # 4. Python version
    python_version = platform.python_version()
    print(f"Python version  : {python_version}")

    print("==================================================")

    # 给个提示
    if cuda_version:
        cu_tag = "cu" + "".join(cuda_version.split("."))
        print("\nSuggested extra index url for pip:")
        print(f"  --extra-index-url https://download.pytorch.org/whl/{cu_tag}")
    else:
        print("\n⚠️ Warning: CUDA version not detected, "
              "please ensure you installed torch with CUDA support.")

if __name__ == "__main__":
    main()
