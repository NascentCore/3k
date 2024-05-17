# cpodoperator

`cpodoperator` 是一个 Kubernetes-native 项目，封装了 Kubeflow 和 Kserve，提供大模型 LLM 微调训练和推理能力。cpodoperator 对
模型和数据集进行封装 ，同时封装了简单易用的无代码微调类型 FineTune，用户只需要指定微调的模型、数据集以及训练的超参数即可以微调模型,
同时 cpodoperator 也提供可自定义化程度更大的 cpodjob 类型，用户可以使用训练任务自定义镜像，自定义训练命令等，使用 cpodoperator 也能够快速部署你的 LLM
应用。

你也可以通过算想云控制台 web 页面使用。

## 类型介绍

模型及数据集

- modelstorage: 模型，cpodoperator 所使用模型对象封装，有以下几种来源：开源模型仓库
- datasetstorage： 数据集，cpodoperator 所使用数据集对象

任务类型
— Finetune: Finetune 是一种无代码微调任务类型,用户可以使用该 finetune 轻松对开源模型做一些微调，用户只需要指定基础模型、数据集以及超参数，同时对于 Lora 微调，可以自动 merge 到处最终模型，
finetune 支持多卡微调，用户只需要指定 GPU 数量和类型，cpodoperator 会自动生成合适的 deepspeed 配置。

- Cpodjob: Cpodjob 是一种更加通用的任务类型，用户可以指定模型、数据集、训练镜像及命令等，Cpodjob 支持分布式训练。
- Training Operator 支持的类型:cpodoperator 也提供对 training operator 的任务类型的支持，用户可以直接使用
  - PytorchJob
  - Mpijobs
  - Tfjobs
  - Elasticdljobs
  - Marsjobs
  - Xgboostjobs
- Inference: Inference 是一种推理部署类型，其封装 kserve 的 inferenceservice, 支持多种主流的推理后端 vLLM、trition 等。

## 使用示例

- 创建 cpodjob 训练任务

```yaml
# Test case for modihand
apiVersion: cpod.cpod/v1beta1
kind: CPodJob
metadata:
  labels:
    app.kubernetes.io/created-by: cpodjob
  name: llama-2-7b
  namespace: cpod
spec:
  # 训练任务镜像
  image: sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/mohe:v1
  # 任务命令
  command:
    - "torchrun"
    - "--nproc_per_node=8"
    - "/workspace/FastChat/fastchat/train/train_mem.py"
    - "--model_name_or_path=/workspace/models"
    - "--data_path=/workspace/FastChat/data/dummy_conversation.json"
    - "--eval_data_path=/workspace/FastChat/data/dummy_conversation.json"
    - "--lazy_preprocess=True"
    - "--bf16=True"
    - "--output_dir=/workspace/modelsaved"
    - "--num_train_epochs=10"
    - "--per_device_train_batch_size=2"
    - "--per_device_eval_batch_size=16"
    - "--gradient_accumulation_steps=4"
    - "--evaluation_strategy=epoch"
    - "--save_strategy=no"
    - "--save_total_limit=10"
    - "--learning_rate=2e-5"
    - "--weight_decay=0.1"
    - "--warmup_ratio=0.04"
    - "--lr_scheduler_type=cosine"
    - "--logging_steps=1"
    - "--fsdp=full_shard auto_wrap"
    - "--fsdp_transformer_layer_cls_to_wrap=LlamaDecoderLayer"
    - "--tf32=True"
    - "--model_max_length=2048"
    - "--gradient_checkpointing=True"
  # 环境变
  env:
    - name: WANDB_MODE
      value: "disabled"
  # 框架类型
  jobType: pytorch
  # 每个 worker需要 GPU的数量
  gpuRequiredPerReplica: 8
  # GPU 类型
  gpuType: NVIDIA-A100-SXM4-40GB
  # worker数量
  workerReplicas: 1
  # 基础模型
  pretrainModelName: llama2-7b
  # 模型挂载路径
  pretrainModelPath: /workspace/models
  # 训练后模型是否上传到 oss
  uploadModel: true
  modelSaveVolumeSize: 100
  modelSavePath: /workspace/modelsaved
```

- 创建 finetune 无代码微调任务

```yaml
apiVersion: cpod.cpod/v1beta1
kind: FineTune
metadata:
  name: finetune-sample
  namespace: cpod
spec:
  # 基础模型
  model: "LLaMA-2-7B"
  # 数据集
  dataset: "llama-2-7b-dataset"
  # 超参数
  hyperParameters:
    n_epochs: "3"
    batch_size: "4"
    learning_rate_multiplier: "5e-5"
```

- 创建 pytorchjob 任务

```yaml
# 单机单卡训练基于gpt3-1.3b 续写模型
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: pytorch-torchrun
  namespace: training-operator
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.cn-beijing.aliyuncs.com/sxwl-ai/modelscope_gpt3_1h1g1dp:latest
              imagePullPolicy: Always
              command:
                - "torchrun"
                - "--nproc_per_node=1"
                - "finetune_poetry.py"
              volumeMounts:
                - name: pretrained-model
                  mountPath: /workspace/nlp_gpt3_text-generation_1.3B
                - name: dataset
                  mountPath: /workspace/chinese-poetry-collection
                - name: saved-model
                  mountPath: /workspace/gpt3_poetry
          volumes:
            - name: pretrained-model
              persistentVolumeClaim:
                claimName: pretrained-model
                readOnly: false
            - name: dataset
              persistentVolumeClaim:
                claimName: dataset
                readOnly: true
            - name: saved-model
              persistentVolumeClaim:
                claimName: saved-model
                readOnly: false
```

- 创建 inference

```yaml
apiVersion: cpod.cpod/v1beta1
kind: Inference
metadata:
  name: infer-d71e9d1b-17df-49de-81f4-11f639f47e97
  namespace: test
spec:
  predictor:
    containers:
      - command:
          - python
          - src/api_demo.py
          - --model_name_or_path
          - /mnt/models
          - --template
          - gemma
        env:
          - name: STORAGE_URI
            value: modelstorage://model-storage-0ce92f029254ff34-public
          - name: API_PORT
            value: "8080"
        image: sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v10
        name: kserve-container
        resources:
          limits:
            cpu: "4"
            memory: 50Gi
            nvidia.com/gpu: "1"
          requests:
            cpu: "4"
            memory: 50Gi
            nvidia.com/gpu: "1"
    nodeSelector:
      nvidia.com/gpu.product: NVIDIA-GeForce-RTX-3090
```

更多示例参考 ./examples 目录

## 开发指南

cpodoperator 项目使用 [kubebuilder](https://book.kubebuilder.io/)，目录在 `./cpodoperator`，以下是添加新类型及 controller 的步骤

1. 创建类型

```bash
kubebuilder create cpodjob --group cpod --version v1beta1 --kind CpodJob
```

2. 修改 API 定义 API 定义在 api 目录，修改对应类型。

3. crd 定义完成，需要生成一些 kubernetes 相关代码。

```bash
make manifests
```

4. 修改 controller 逻辑， controller 逻辑在 internal/controller 目录，每个 controller 对应一个 go 文件。

5. 编译发布

```bash

make docker-build && make docker-push

```
