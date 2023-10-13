# Bert

Use Bert training written in PyTorch+DeepSpeed, used to test 3k.

Based on:

https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert.py

https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert_ds.py

## Training code

First [install `miniconda`](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html);
alternatively, you may opt to install everything for your own user locally.

```
# Install system packages
sudo apt update
sudo apt install libopenmpi-dev ninja-build

# Create conda environment: bert_deepspeed
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
python train_bert.py  --local_rank 0 --checkpoint_dir ./experiments

# Run bert training on all local GPUs
deepspeed train_bert_ds.py --checkpoint_dir ./experiments

# 支持本机任意多卡运行
# 只有编号为0,2,3的GPU对程序是可见的，在代码中gpu[0]指的是第0块，
# gpu[1]指的是第2块，gpu[2]指的是第3块
CUDA_VISIBLE_DEVICES=0,2,3 deepspeed train_bert_ds.py --checkpoint_dir ./experiments

# 多机多卡(以3机24卡为例)
# 要求：配置hostfile文件；每台机器上代码存储路径都一致，
# 并且对应的conda虚拟环境也一致;各台机器间ssh免密登陆；有.deepspeed_env文件；
# 注：当在多个节点上进行训练时，我们发现支持传播用户定义的环境变量非常有用。
# 默认情况下，DeepSpeed 将传播所有设置的 NCCL 和 PYTHON 相关环境变量。
# 如果您想传播其它变量，可以在名为 .deepspeed_env 的文件中指定它们，
# 该文件包含一个行分隔的 VAR=VAL 条目列表。
# DeepSpeed 启动器将查找你执行的本地路径以及你的主目录（~/）
deepspeed --hostfile=hostfile  --master_port 60000 train_bert_ds.py \
    --checkpoint_dir ./experiments
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
