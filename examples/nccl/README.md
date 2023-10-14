# NCCL

## Building NCCL

[Instructions](https://github.com/NVIDIA/nccl#install)

```
cd nccl
make -j src.build

# Build deb package
sudo apt-get install devscripts
make pkg.debian.build

# Install libnccl2
sudo dpkg -i build/pkg/deb/libnccl2_2.18.6-1+cuda11.7_amd64.deb
sudo dpkg -i build/pkg/deb/libnccl-dev_2.18.6-1+cuda11.7_amd64.deb
```

## NCCL-TESTS

* An in-place operation uses the same buffer for its output as was used to
  provide its input. An out-of-place operation has distinct input and output
  buffers [ref](https://github.com/NVIDIA/nccl/issues/12).

## NCCL PyTorch tests

* Single GPU
  ```
  # Runs a single worker process on the localhost
  CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 MASTER_PORT=29500 MASTER_ADDR=127.0.0.1 python dist_nccl_demo.py
  ```
* Distributed
  Start N workers, can be on localhost or across multiple hosts, forms distributed collective communication:
  ```
  WORLD_SIZE=n RANK=0 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py
  WORLD_SIZE=n RANK=1 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py 
  WORLD_SIZE=n RANK=2 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py 
  ...
  WORLD_SIZE=n RANK=n-1 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py  
  ```

## NOTES

* NCCL always use chains [LL-128](https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990)
  for intra-node all reduce, see [NVIDIA/nccl/issues/919](https://github.com/NVIDIA/nccl/issues/919)
