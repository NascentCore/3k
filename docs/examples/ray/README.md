## 编译 vllm ray 镜像

该镜像是 rayService 的 HEAD 和 Worker 镜像，包含 ray 和 vllm 以及一个 APPlication

```bash
docker build -t sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/vllm-ray:latest -f Dockerfile .
```

## 生成 ray serve 配置文件

## 使用 locust 压力测试

```bash
locust -f locustfile.py --host=http://10.10.10.10:8000
```
