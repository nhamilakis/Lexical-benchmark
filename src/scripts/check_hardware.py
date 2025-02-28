#!/usr/bin/env python
import multiprocessing
import platform

import torch

print("\n=== Python Hardware Detection ===")
print(f"Python Version: {platform.python_version()}")
print(f"PyTorch Version: {torch.__version__}")
print("\nCPU Information:")
print(f"Number of CPU cores: {multiprocessing.cpu_count()}")
print(f"Number of CPU threads available to PyTorch: {torch.get_num_threads()}")

if torch.cuda.is_available():
    print("\nGPU Information:")
    print("CUDA Available: Yes")
    print(f"CUDA Version: {torch.version.cuda}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {gpu.name}")
        print(f"  Memory: {gpu.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {gpu.major}.{gpu.minor}")
else:
    print("\nNo CUDA-capable GPU detected")

# Simple tensor operations to verify GPU functionality
if torch.cuda.is_available():
    print("\nVerifying GPU functionality:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print(f"GPU {i}: Matrix multiplication test successful")
