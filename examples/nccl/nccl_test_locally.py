#!/bin/env python
#
# An minimal use case of NCCL by using pytorch
#
# Usage: python nccl_test_locally.py
# -------------------------------------------------------------------

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

# x1, x2 values should be identical now
print(x1)
print(x2)
