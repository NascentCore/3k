import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

dist.init_process_group(backend="nccl")

print("LOCAL_RANK:{}".format(os.environ["LOCAL_RANK"]))
print("GLOBAL_RANK:{}".format(os.environ["RANK"]))
print("GROUP_RANK:{}".format(os.environ["GROUP_RANK"]))
print("ROLE_RANK:{}".format(os.environ["ROLE_RANK"]))
print("LOCAL_WORLD_SIZE:{}".format(os.environ["LOCAL_WORLD_SIZE"]))

tensor = torch.ones(1,device="cuda:{}".format(os.environ["LOCAL_RANK"]))
dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
print('ALL_REDUCE_SUM_OF_ALL_WORKER:{}'.format(tensor))