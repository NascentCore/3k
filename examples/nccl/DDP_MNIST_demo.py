#!/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # from https://arxiv.org/abs/1202.2745
        self.module = nn.Sequential(nn.Conv2d(1, 20, 4),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(20, 40, 5),
                                    nn.MaxPool2d(3),
                                    nn.Flatten(),
                                    nn.Linear(40 * 3 * 3, 150),
                                    nn.Linear(150, 10))

    def forward(self, x):
        return self.module(x)

def partition_dataset(global_batch_size):
    class DataPartitioner(object):
        def __init__(self, data, sizes):
            self.data = data
            self.partitions = []
            data_len = len(data)
            # All the "indexes" in the world are same
            # because of the same seed setted by "torch.manual_seed"
            indexes = torch.randperm(data_len)
            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

        def use(self, partition):
            class Partition(object):
                def __init__(self, data, index):
                    self.data = data
                    self.index = index

                def __len__(self):
                    return len(self.index)

                def __getitem__(self, index):
                    data_idx = self.index[index]
                    return self.data[data_idx]

            return Partition(self.data, self.partitions[partition])

    class MyTransform:
        def __init__(self):
            pass

        def __call__(self, img):
            img = transforms.functional.rotate(img, -90)
            img = transforms.functional.hflip(img)
            img = img.map_(img, lambda a, b: 1.0 - a)
            return img

    dataset = datasets.MNIST("./data", train=True, download=True,
                    transform = transforms.Compose([transforms.PILToTensor(),
                            transforms.Resize(29),
                            transforms.ConvertImageDtype(torch.float32),
                            MyTransform()
                    ]))

    world_size = dist.get_world_size()
    local_batch_size = int(global_batch_size / float(world_size))

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partitions = DataPartitioner(dataset, partition_sizes)
    one_partition = partitions.use(dist.get_rank())

    train_set = torch.utils.data.DataLoader(one_partition,
                                            batch_size=local_batch_size,
                                            shuffle=True)
    return train_set, local_batch_size

def run(rank, size):
    torch.manual_seed(1234)
    global_batch_size = 512
    train_set, local_batch_size = partition_dataset(global_batch_size)

    model = Net().cuda()
    ddp_model = DDP(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(10):
        epoch_loss = 0.0
        for (data, target) in train_set:
            (data, target) = (data.cuda(), target.cuda())

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Rank ", dist.get_rank(), ", epoch ",
              epoch, ": ", epoch_loss / local_batch_size)

print("RANK = " + os.environ["RANK"])
print("WORLD_SIZE = " + os.environ["WORLD_SIZE"])
print("MASTER_ADDR = " + os.environ["MASTER_ADDR"])
print("MASTER_PORT = " + os.environ["MASTER_PORT"])
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,P2P,NET,COLL,GRAPH"
os.environ["NCCL_NET"] = "Socket"
os.environ["NCCL_NET_GDR_LEVEL"] = "SYS"

dist.init_process_group(backend="nccl")
run(dist.get_rank(), dist.get_world_size())
