# 多机多卡示例

本示例来自 [pytorch example 库](https://github.com/pytorch/examples)，
用于测试多机多卡的简单 torch.nn.liner 训练任务。

## 裸机测试

在需要加入测试的节点上执行，这里我们在 node1 和 node2 两个节点上进行测试，多卡之间通信走 TCP 网络。

在 node1 和 node2 上都执行：

```bash

# 创建conda虚拟环境
conda create -c pytorch -c nvidia -n pytorch+cuda_11-7 pytorch==2.0.1\
torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7

conda activate pytorch+cuda_11-7

# 设置NCCL相关环境变量，保证NCCL通信正常(可能出现的问题: 单个节点有IB网卡、探测使用到不正确的网卡)
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1

```

在 node1 上执行

```bash
# nnodes: 参与训练GPU的数量
# nproc_per_node: 每个节点训练任务的worker数量
# node-rank: 节点标志
# rdzv-endpoint: rdzv 地址
# 50 10: 进行50 epochs训练，snapshot间隔时间
torchrun  --nnodes=2  --nproc_per_node=4 --node-rank=0 \
--rdzv-endpoint=214.2.5.4:60002 --rdzv-backend=c10d  \
--rdzv_id=123  test.py  50 10
```

在 node2 上执行

```bash
torchrun  --nnodes=2  --nproc_per_node=4 --node-rank=1 \
--rdzv-endpoint=214.2.5.4:60002 --rdzv-backend=c10d  \
--rdzv_id=123  test.py  50 10
```

观察输出

node1

```text
[GPU3] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 46 | Batchsize: 32 | Steps: 8

```

node2

```text
...
Epoch 40 | Training snapshot saved at snapshot.pt
[GPU4] Epoch 41 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 42 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 43 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 45 | Batchsize: 32 | Steps: 8
...
```

## k8s 测试

执行`kubectl create -f ./pytorch_multinode_linertrain.yaml`
