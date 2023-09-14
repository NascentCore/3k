#!/bin/env python
import torch
import os
import time

os.environ["NCCL_DEBUG"] = "INFO"

x1 = torch.tensor([i for i in range(1024)], dtype=torch.float32, device=torch.device("cuda:0"))
x2 = torch.tensor([1 for i in range(1024)], dtype=torch.float32, device=torch.device("cuda:1"))

print(x1)
print(x2)

t = time.perf_counter()
torch.cuda.nccl.all_reduce([x1,x2])
print(f'coast:{time.perf_counter() - t:.8f}s')

print(x1)
print(x2)
