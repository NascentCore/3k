# GPT3

This sample is based on [ModeScope GPT3](
https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary).

## The original samples
```
torchrun finetune_poetry.py
# N 为模型并行度
torchrun --nproc_per_node $N finetune_dureader.py
```

## SXWL adapted version
Install ModelScope library core framework
```
https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install modelscope
```
Env vars
```
export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=SYS NCCL_IB_CUDA_SUPPORT=1 NCCL_DEBUG_SUBSYS=ALL NCCL_IB_GDR_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_NET=IB
```
Single node
```
torchrun --standalone --nproc_per_node 8 --nnodes=1 --node_rank=0 --master_addr= --master_port= modelscope_sxwl_gpt3.py
torchrun --standalone --nproc_per_node 8 --nnodes=1 modelscope_sxwl_gpt3.py
```
