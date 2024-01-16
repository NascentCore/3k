# Docker

基础镜像dockerfile文件名以 `<imagename>.Dockerfile` 命名，合并至 main 分支后将自动构建，镜像以 `<imagename>` 命名，以 `v$(date +%Y-%m-%d)` 为tag

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
