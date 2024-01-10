# llama2-pretrain
为了验证3k平台支持预训练能力，我们按照如下步骤来完成验证

1. llama2-7b-fine-tune-bare-mental-single-gpu 1h1g
2. llama2-7b-fine-tune-bare-mental-multi-gpu 1h8g
3. llama2-7b-fine-tune-bare-mental-multi-gpu-full-params 1h8g 等效预训练
4. llama2-7b-fine-tune-3k 等效预训练在3k平台
5. llama2-70b 模型扩大至70b，重复验证

## 当前进度
[x] llama2-7b-fine-tune-bare-mental-single-gpu
[ ] llama2-7b-fine-tune-bare-mental-multi-gpu

## 问题
1h8g 运行报错，在A100机器上，复现方法
```bash
cd /data/chenshu/3k/examples/llama2-pretrain
source venv/bin/activate
python3 7b-fine-tune-bare-mental-multi-gpu.py
```
