# NCCL

## Single GPU
```
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=3 RANK=2 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py
```

## 2 GPUs
```
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=3 RANK=0 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py
CUDA_VISIBLE_DEVICES=1 WORLD_SIZE=3 RANK=1 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py
```

## Distributed

启动n个进程（跨多台物理机、只要网络连通即可）搞集合通信
```
WORLD_SIZE=n RANK=0 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py
WORLD_SIZE=n RANK=1 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py 
WORLD_SIZE=n RANK=2 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py 
...
WORLD_SIZE=n RANK=n-1 MASTER_PORT=29500 MASTER_ADDR=192.168.0.206 ./dist_nccl_demo.py  
```

## Build NCCL
```
cd nccl
make

# Needed to build deb package
sudo apt-get install devscripts

# Install libnccl2
sudo dpkg -i build/pkg/deb/libnccl2_2.18.6-1+cuda11.7_amd64.deb
sudo dpkg -i build/pkg/deb/libnccl-dev_2.18.6-1+cuda11.7_amd64.deb
```
