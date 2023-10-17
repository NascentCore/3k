#!/bin/env python
import os
import torch
import torch.distributed as dist

print("RANK = " + os.environ['RANK'])
print("WORLD_SIZE = " + os.environ['WORLD_SIZE'])
print("MASTER_ADDR = " + os.environ['MASTER_ADDR'])
print("MASTER_PORT = " + os.environ['MASTER_PORT'])
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = "INIT,P2P,NET,COLL,GRAPH"
os.environ['NCCL_NET'] = 'Socket'
os.environ['NCCL_NET_GDR_LEVEL'] = 'SYS'

print(2)
dist.init_process_group(backend="nccl")

print(3)

rank = int(os.environ['RANK'])
x1 = torch.tensor([rank + 1] * 16, dtype=torch.float32).cuda()
print(x1)

dist.all_reduce(x1, op=dist.ReduceOp.SUM)
print(x1)
