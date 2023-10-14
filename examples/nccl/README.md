# NCCL

## Single GPU
```
# Runs a single worker process on the localhost
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 MASTER_PORT=29500 MASTER_ADDR=127.0.0.1 python dist_nccl_demo.py
```

## Distributed

Start N workers, can be on localhost or across multiple hosts, forms distributed collective communication:
```
WORLD_SIZE=n RANK=0 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py
WORLD_SIZE=n RANK=1 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py 
WORLD_SIZE=n RANK=2 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py 
...
WORLD_SIZE=n RANK=n-1 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py  
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
