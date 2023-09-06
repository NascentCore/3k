import torch

# python torch_check_cuda.py
# Prints the info about the GPU
print("Is CUDA available:", torch.cuda.is_available(),
        "\nCurrent device:", torch.cuda.current_device(),
        "\nDevice[0] name:", torch.cuda.get_device_name(0))
