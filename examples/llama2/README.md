# 概述
该项目用于将llama2 7b模型分别在物理机、docker、k8s上用于验证模型的预训练

## 创建基于deepspeed的llama2分布式预训练代码
以下代码测试在80G A100上运行成功

## 容器镜像构建
```shell
docker build . -f Dockerfile.20231013 -t "llama2_demo:$(date +%F)"
docker login --username=eng@nascentcore.ai registry.ap-southeast-1.aliyuncs.com
docker tag "llama2_demo:$(date +%F)" "registry.ap-southeast-1.aliyuncs.com/sxwl-ai/llama2_demo:$(date +%F)"
docker push "registry.ap-southeast-1.aliyuncs.com/sxwl-ai/llama2_demo:$(date +%F)"
```
