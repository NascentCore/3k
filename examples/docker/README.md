# Docker

## Build CUDA base image

CUDA base image include CUDA, OpenMPI, and SSHD for starting a distributed MPI training job

```
docker build -f cuda_base.Dockerfile . -t cuda_base
docker tag cuda_base:latest registry.cn-hangzhou.aliyuncs.com/sxwl-ai/common:v2023-10-16-00
docker login --username=eng@nascentcore.ai registry.cn-hangzhou.aliyuncs.com
docker push registry.cn-hangzhou.aliyuncs.com/sxwl-ai/common:v2023-10-16-00
```
