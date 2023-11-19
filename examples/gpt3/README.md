# 基于魔搭 GPT-3 预训练生成模型-中文-1.3B 的 ML 模型二次训练

This sample is adapted from [ModeScope GPT3](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary).

## 编译镜像

```
sudo apt install git-lfs
git lfs install

# Download pretrained model
git clone https://modelscope.cn/damo/nlp_gpt3_text-generation_1.3B.git
cd nlp_gpt3_text-generation_1.3B
# This removes git metadata, and almost halves the size
rm -rf .git

# Download dataset
git clone http://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection.git
cd chinese-poetry-collection
rm -rf .git

# Make sure the directory layout is as follows
[0:29:25] yzhao:gpt3 git:(main) $ ls
chinese-poetry-collection  Dockerfile  finetune_dureader.py  finetune_poetry.py  nlp_gpt3_text-generation_1.3B  README.md

# Build docker image
docker build . -t modelscope_gpt3:v002
docker tag modelscope_gpt3:v002 registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002
docker push registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002

```

## 裸机中容器运行

执行以下命令

```
docker run -it --gpus=all registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modelscope_gpt3:v002 bash
root@a2bd9b4ad983:/workspace# ls
finetune_dureader.py  finetune_poetry.py  nlp_gpt3_text-generation_1.3B
root@a2bd9b4ad983:/workspace# torchrun finetune_poetry.py
root@a2bd9b4ad983:/workspace# torchrun --nproc_per_node=1 finetune_poetry.py
```

## 通过 pytorchjob 运行

运行`ptjob_gpt3_1.3b_1h1g.yaml`单机单卡训练任务，此任务最小需要 32G 显存。如果单卡内存小于 32G 显存如 3090，则可以运行单机多卡的 pytorch job `ptjob_gpt3_1.3b_1h8g.yaml`。

```
kubectl create -f ptjob_gpt3_1.3b_1h1g.yaml
```

观察对应 pytorch job 和 worker 对应的 pod 是否运行成功

```
$ kubectl -n training-operator get pytorchjob
NAME               STATE     AGE
pytorch-torchrun   Created   47s

kubectl -n training-operator get pods -owide
NAME                                 READY   STATUS    RESTARTS   AGE   IP               NODE      NOMINATED NODE   READINESS GATES
pytorch-torchrun-worker-0            1/1     Running   0          21s   10.233.103.133   worker2   <none>           <none>
```

pod 运行在 worker2，进入 对应 nvidia-driver 容器中，调用 nvidia-smi,观察 gpu 是否运行,

```

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
