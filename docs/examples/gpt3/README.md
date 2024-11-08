# 基于魔搭 GPT-3 预训练生成模型-中文-1.3B 的 ML 模型二次训练

This sample is adapted from [ModeScope GPT3](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary).

## 编译镜像

```bash
sudo apt install git-lfs
git lfs install

# Download pretrained model
git clone https://modelscope.cn/damo/nlp_gpt3_text-generation_1.3B.git
cd nlp_gpt3_text-generation_1.3B
# This removes git metadata, and almost halves the size
rm -rf .git

# modify and replace megatron configuration.json
cp ./configuration.json  nlp_gpt3_text-generation_1.3B/

# Download dataset
git clone http://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection.git
cd chinese-poetry-collection
rm -rf .git

# Make sure the directory layout is as follows
[0:29:25] yzhao:gpt3 git:(main) $ ls
chinese-poetry-collection  Dockerfile  finetune_dureader.py
finetune_poetry.py  nlp_gpt3_text-generation_1.3B  README.md

# Build docker image
docker build . -t modelscope_gpt3:v002
docker tag modelscope_gpt3:v002 registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002
docker push registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002

```

## 单机单卡和多机多卡训练

### 裸机中容器运行

执行以下命令

```bash
# 需要扩大容器默认共享内存大小，默认是64M 通过 --shm-size=X设置，
# 也可以直接设置--ipc=host
docker run -it --gpus=all --ipc=host \
registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002 bash

root@a2bd9b4ad983:/workspace# ls
finetune_dureader.py  finetune_poetry.py  nlp_gpt3_text-generation_1.3B
root@a2bd9b4ad983:/workspace# torchrun finetune_poetry.py
root@a2bd9b4ad983:/workspace# torchrun --nproc_per_node=1 finetune_poetry.py
```

## 通过 pytorchjob 运行

运行`ptjob_gpt3_1.3b_1h1g.yaml`单机单卡训练任务，此任务最小需要 32G 显存。
如果单卡内存小于 32G 显存如 3090，则可以运行单机多卡的 pytorch job `ptjob_gpt3_1.3b_1h8g.yaml`。

```bash
kubectl create -f ptjob_gpt3_1.3b_1h1g.yaml
```

观察对应 pytorch job 和 worker 对应的 pod 是否运行成功

```bash
$ kubectl -n training-operator get pytorchjob
NAME               STATE     AGE
pytorch-torchrun   Created   47s

kubectl -n training-operator get pods -owide
NAME                                 READY   STATUS    RESTARTS   AGE   IP               NODE      NOMINATED NODE   READINESS GATES
pytorch-torchrun-worker-0            1/1     Running   0          21s   10.233.103.133   worker2   <none>           <none>
```

pod 运行在 worker2，进入 对应 nvidia-driver 容器中，调用 nvidia-smi,观察 gpu 是否运行,

```bash

$ kubectl exec -it nvidia-driver-daemonset-9l95r  -n gpu-operator  -- bash

# pod 容器执行
$ nvidia-smi
...
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     28352      C   /usr/bin/python3                          32798MiB |
+---------------------------------------------------------------------------------------+
```

## 多机多卡训练任务

### 裸金属运行

```bash

# Step1: 安装conda

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
# 安装完成后退出bash重新进入


# Step2: 创建 conda env，安装pytorch和魔搭库等依赖库
conda create -n modelscope python=3.8
conda activate modelscope
# 这里使用pip安装，不要用conda安装
pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple   torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple tensorboard transformers ninja modelscope megatron_util jieba -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install -U datasets

# Step3: 准备代码、数据集和模型
cd 3k/examples/gpt3/

git clone https://modelscope.cn/damo/nlp_gpt3_text-generation_1.3B.git
cd nlp_gpt3_text-generation_1.3B
rm -rf .git
cd ..

git clone http://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection.git
cd chinese-poetry-collection
rm -rf .git
cd ..
# 修改finetune_poetry.py模型和数据集的加载路径
vim finetune_poetry.py
# line 18 ->  train_dataset = MsDataset.load("./chinese-poetry-collection/train.csv").remap_columns({'text1': 'src_txt'})
# line 19 ->  eval_dataset = MsDataset.load("./chinese-poetry-collection/test.csv").remap_columns({'text1': 'src_txt'})

# 配置NCCL变量，以下使用TCP
export NCCL_SOCKET_IFNAME=bond0 # 网卡根据节点环境修改
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_FAMILY=AF_INET

# 如果要使用IB 使用下面的环境变量
# $unset ${!NCCL*}
# export NCCL_DEBUG=INFO
# export NCCL_NET=IB
# export NCCL_SOCKET_FAMILY=bond0


# 配置ssh config 和 etc hosts 文件


# 以上所有配置在worker4和worker5 都需要执行
 #######################################

# 在worker4 中执行
python -m torch.distributed.run  --nnodes=2  --nproc_per_node=8 --node-rank=0 --rdzv-endpoint=214.2.5.4:60002 --rdzv-backend=c10d  --rdzv_id=123  finetune_poetry_host.py

#  在worker5 上执行
python -m torch.distributed.run  --nnodes=2  --nproc_per_node=8 --node-rank=1 --rdzv-endpoint=214.2.5.4:60002 --rdzv-backend=c10d  --rdzv_id=123  finetune_poetry_host.py 2>&1 | tee log.txt

```

### 通过 pytorchjob 运行多机多卡

```bash
kubectl create -f ptjob_gpt3_1.3b_2h16g.yaml
```
