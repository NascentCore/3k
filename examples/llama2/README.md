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

Environment variables
```shell
export NCCL_DEBUG=INFO NCCL_P2P_LEVEL=SYS NCCL_IB_CUDA_SUPPORT=1 NCCL_DEBUG_SUBSYS=ALL \
NCCL_IB_GDR_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_NET=IB NCCL_SOCKET_IFNAME=ibs1 NCCL_IB_HCA=mlx5_
```

`NCCL_IB_GDR_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_NET=IB NCCL_SOCKET_IFNAME=ibs1 NCCL_IB_HCA=mlx5_`
configures IB & RDMA related behaviors.

```
# On master node, get IP with `ip a`; --node_rank=0 means this is master
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=<node-count> \
    --node_rank=0 --master_addr=<local-ip> --master_port=<port> \
    llama2_demo.py

# All other nodes needs to set --node_rank accordingly
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=<node-count> \
    --node_rank=<node-rank> --master_addr=<master-ip> --master_port=<port> \
    llama2_demo.py
```
### Multiple nodes launched by *mpirun*
```
export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=SYS NCCL_IB_CUDA_SUPPORT=1 \
    NCCL_DEBUG_SUBSYS=ALL NCCL_IB_GDR_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS \
    NCCL_NET_GDR_READ=1 NCCL_NET=IB
mpirun -np 8 --nooversubscribe -H worker4:4,worker5:4 -x MASTER_ADDR=worker4 \
    -x MASTER_PORT=29501 -x PATH -bind-to none -map-by slot -mca pml ob1 \
    -mca btl ^openib \
    python3 llama2_demo.py
```

## Docker
```shell
docker build . -f Dockerfile.20231013 -t "llama2_demo:$(date +%F)"
docker login --username=eng@nascentcore.ai registry.ap-southeast-1.aliyuncs.com
docker tag "llama2_demo:$(date +%F)" "registry.ap-southeast-1.aliyuncs.com/sxwl-ai/llama2_demo:$(date +%F)"
docker push "registry.ap-southeast-1.aliyuncs.com/sxwl-ai/llama2_demo:$(date +%F)"
```

## Notes

### cpu_adam cuda warning
```
cpu_adam cuda is missing or is incompatible with installed torch, only cpu ops can be compiled!
```
See [DeepSpeed Issue/3613](https://github.com/microsoft/DeepSpeed/issues/3613#issuecomment-1581104500)

### urllib3 chardet supported version

If you see warning message:
```
urllib3 (1.25.10) or chardet (3.0.4) doesn't match a supported version!
```
Run `python3 -m pip install -U urllib3 requests` to upgrade relevant packages.

### Env vars

```
Env vars:
- NCCL_DEBUG=INFO
- NCCL_NET=IB
- NCCL_P2P_LEVEL=SYS  # Try to enforce P2P anywhere.
- NCCL_SOCKET_IFNAME=
- NCCL_IB_HCA=
- NCCL_IB_CUDA_SUPPORT=1  # Try to enforce GDR.
- NCCL_NET_GDR_LEVEL=SYS
- NCCL_IB_GDR_LEVEL=SYS
- NCCL_NET_GDR_READ=1
- NCCL_DEBUG_SUBSYS=ALL

Note:
- "NCCL_P2P_DISABLE=1" should not be set.
- "NCCL_P2P_DIRECT_DISABLE=1" should not be set.
- ibdev2netdev to check NCCL_SOCKET_IFNAME (rhs) and NCCL_IB_HCA (lhs)
- "NCCL_COLLNET_ENABLE=1" may need to be set.
- "NCCL_COLLNET_NODE_THRESHOLD=17" may need to be set.
- NCCL version
```
