# Bert

Bert training written in PyTorch+DeepSpeed.
Based on the examples in the official DeepSpeed repo:
* https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert.py
* https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert_ds.py

## Prerequisite

```
sudo apt-get install libopenmpi-dev
```

## Training code

```
# Install system packages
# openmpi-dev needed by mpi4py, ninja-build used by PyTorch, pdsh used by deepspeed
sudo apt update
sudo apt install libopenmpi-dev ninja-build pdsh

# Create conda environment: bert_deepspeed
# First, [install `miniconda`](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)
# You may skip this step to install everything in the system directly
conda create -n bert_deepspeed python==3.11
conda activate bert_deepspeed

git clone git@github.com:NascentCore/3k.git
cd 3k/examples/bert_deepspeed

pip install -r requirements.txt

# Check deepspeed installation
# It's possible that installed packages' executable files are not configured
# correctly in PATH env var, you need to find ds_report: find ~/ -name ds_report
ds_report

# Run bert training on 1 local GPU
python train_bert.py  --local_rank 0 --checkpoint_dir experiments

# Run distributed bert training on all local GPUs
# It's possible that installed packages' executable files are not configured
# correctly in PATH env var, you need to find ds_report: find ~/ -name deepspeed
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO
deepspeed train_bert_ds.py --checkpoint_dir experiments

# Use only specified GPUs
CUDA_VISIBLE_DEVICES=0,2,3 deepspeed train_bert_ds.py --checkpoint_dir experiments
```

### Distributed training on mulitple nodes

Configuration [hostfile](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)
hostnames (or SSH aliases)

```
worker-1 slots=4
worker-2 slots=4
```

Copy hostfile onto all hosts listed above, in the same path, then launch
deepspeed distributed training:
```
# Multi-host: https://www.deepspeed.ai/getting-started/
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO
deepspeed --hostfile=hostfile  --master_port 60000 train_bert_ds.py \
    --checkpoint_dir experiments
```

## Container

When the training PyTorch code is updated, run the following to build and push
new image version.

```
docker build . -t swr.cn-east-3.myhuaweicloud.com/sxwl/bert:<version>
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/bert:<version>
```

If you need to update the `bert-base` image:

```
docker build . -f base.Dockerfile -t swr.cn-east-3.myhuaweicloud.com/sxwl/bert-base
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/bert-base
```
