# 裸金属训练

1. 安装 cuda12.2
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

2. 创建虚拟环境并安装依赖
```bash
conda create --name myenv python=3.9
conda activate myenv
pip install torch transformers datasets deepspeed tensorboard
pip install accelerate -U
```

3. 单机多卡训练
```bash
torchrun --nproc_per_node=4 train.py
```

4. 多机多卡训练
- 确保多机 cuda 版本以及 torch 版本一致
- A 机器启动训练脚本
```bash
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC

export MASTER_ADDR=214.2.5.3
export MASTER_PORT=23455
export NODE_RANK=0
export WORLD_SIZE=8

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py
```

- B 机器启动训练脚本
```bash
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC

export MASTER_ADDR=214.2.5.3
export MASTER_PORT=23455
export NODE_RANK=1
export WORLD_SIZE=8

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py
```