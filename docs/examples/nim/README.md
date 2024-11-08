
运行镜像，该镜像不包含模型，需要将模型挂载到 /mnt/models 目录下

```bash
 docker run -it --rm --gpus '"device=1"' --ipc=host -p 8080:8080 -p 8000:8000 -v /data2/dg/models/meta-llama-3.1-8b-instruct:/mnt/models sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/sxwl-nim:v1.2
```

运行镜像，该镜像包含模型，模型在镜像中

```bash
docker run -it --rm --gpus '"device=1"' --ipc=host -p 8080:8080 -p 8000:8000 sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/sxwl-nim-with-model:v1.2
```
