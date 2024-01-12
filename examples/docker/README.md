# Docker

## Build CUDA base image

CUDA base image include CUDA, OpenMPI, and SSHD for starting a distributed MPI training job

```
docker build -f cuda_base.Dockerfile . -t sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/cuda-base:v2024-01-12-00 .
docker push sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/cuda-base:v2024-01-12-00
```

## Build Torch base image
Torch base image include CUDA, OFED, OpenSM, net-tools, iputils-ping, git, vim

```
docker build -f torch_base.Dockerfile -t sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/torch-base:v2024-01-12-00 .
docker push sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/torch-base:v2024-01-12-00
```
