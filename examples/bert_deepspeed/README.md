# 概述

bert_deepspeed项目用于将 roberta（类 bert）模型分别在物理机、docker、k8s上用于验证模型的预训练

## 运行训练代码

以下代码经测试在 3090 上运行成功

```
# 配置虚拟环境
conda create -n bert_deepspeed python==3.11
conda activate bert_deepspeed
git clone git@github.com:NascentCore/3k.git
cd 3k/examples/bert_deepspeed
pip install -r requirements.txt

# 单卡
python train_bert.py  --local_rank 0 --checkpoint_dir ./experiments

# 多卡使用本地全部 GPU
deepspeed train_bert_ds.py --checkpoint_dir ./ds_experiments

# 支持本机任意多卡运行
# 只有编号为0,2,3的GPU对程序是可见的，在代码中gpu[0]指的是第0块，gpu[1]指的是第2块，gpu[2]指的是第3块
export CUDA_VISIBLE_DEVICES=0,2,3;deepspeed train_bert_ds.py --checkpoint_dir ./ds_experiments
```
