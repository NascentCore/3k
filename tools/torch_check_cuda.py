import torch

# python torch_check_cuda.py
# Prints the info about the GPU
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
