1. 创建测试pod 

```bash

kubectl create -f pod.yaml

```

2. 查看pod日志

```bash

kubectl -n gpu-operator logs exporter

```

![alt text](image.png)