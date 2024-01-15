# llama2-full-param-finetune
为了验证3k平台支持预训练能力，我们按照如下步骤来完成验证

1. llama2-7b-fine-tune-full-param(bare mental)
2. llama2-7b-fine-tune-full-param(3k)
3. llama2-70b

## 说明
代码主要来源是 [llama-recipes](https://github.com/facebookresearch/llama-recipes) 官方的示例文档库

## 安装依赖
注意: 不加代理可能很慢

```bash
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```

## 模型和数据集
model: [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)

dataset: [alpaca_data](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json)

dataset需要下载到执行目录下的`src/llama_recipes/datasets/`

## 启动命令
根据实际情况调整GPU数量设置、checkpoint目录、模型目录、模型输出目录
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export FINE_TUNE_NODES=1
export FINE_TUNE_PERNODE_PROCS=4

torchrun --nnodes $FINE_TUNE_NODES --nproc_per_node $FINE_TUNE_PERNODE_PROCS full-param/main.py \
    --enable_fsdp \
    --save_optimizer \
    --dist_checkpoint_root_folder /data/chenshu/model_checkpoints \
    --dist_checkpoint_folder multi-gpu-full-param-fine-tune \
    --model_name /data/models/hf_models/llama-2-7b/ \
    --dataset alpaca_dataset \
    --output_dir /data/chenshu/model/7b_PEFT_multi_full_param_gpu/model
```
