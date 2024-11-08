#!/bin/env python
#
# An minimal use case of NCCL by using pytorch
#
# Usage: python nccl_test_locally.py
# -------------------------------------------------------------------
# TODO: Add tests for "reduce", "broadcast", "all_gather", "reduce_scatter" other NCCL wrapper APIs
# in PyTorch.

import torch
import os
import time
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "INFO"

arrs = []
for i in range(torch.cuda.device_count()):
    cuda_dev_name = 'cuda:{}'.format(i)
    x = torch.tensor([i for j in range(128)], dtype=torch.float32,
                     device=torch.device(cuda_dev_name))
    arrs.append(x)

print(arrs)

t = time.perf_counter()
torch.cuda.nccl.all_reduce(arrs,
                           # Change to other operator for testing
                           op=dist.ReduceOp.SUM)

print(f'coast:{time.perf_counter() - t:.8f}s')

# x1, x2 values should be identical now
print(arrs)
