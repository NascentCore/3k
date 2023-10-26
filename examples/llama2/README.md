# LLaMA2

LLaMA2 model training.

## Single host

Use torch's own launcher:
```
NCCL_DEBUG=DEBUG NCCL_P2P_LEVEL=LOC NCCL_NET=IB NCCL_NET_GDR_LEVEL=SYS \
NCCL_SHM_DISABLE=1 \
python3 -m torch.distributed.launch --nproc_per_node=8 llama2_demo.py
```

## Multiple hosts
```
# On master node, get IP with `ip a`; --node_rank=0 means this is master
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=<node-count> --node_rank=0 \
    --master_addr=<local-ip> --master_port=<port> llama2_demo.py

# All other nodes needs to set --node_rank accordingly
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=<node-index> \
    --master_addr=<local-ip> --master_port=<port> llama2_demo.py
```

## Docker
```shell
docker build . -f Dockerfile.20231013 -t "llama2_demo:$(date +%F)"
docker login --username=eng@nascentcore.ai registry.ap-southeast-1.aliyuncs.com
docker tag "llama2_demo:$(date +%F)" "registry.ap-southeast-1.aliyuncs.com/sxwl-ai/llama2_demo:$(date +%F)"
docker push "registry.ap-southeast-1.aliyuncs.com/sxwl-ai/llama2_demo:$(date +%F)"
```
