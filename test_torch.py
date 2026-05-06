import torch
print("Torch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")