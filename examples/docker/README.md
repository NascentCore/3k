# Docker

```
docker build -f cuda_base.Dockerfile . -t cuda_base
docker tag cuda_base:latest registry.cn-hangzhou.aliyuncs.com/sxwl-ai/common:v2023-10-16-00
docker login --username=eng@nascentcore.ai registry.cn-hangzhou.aliyuncs.com
docker push registry.cn-hangzhou.aliyuncs.com/sxwl-ai/common:v2023-10-16-00
```
