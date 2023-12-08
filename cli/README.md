# 3kctl
3kctl是3k平台的命令行工具

## 3kctl deploy

## 3kctl download
### 测试用例
在魔搭上找到自己需要的模型，例如https://modelscope.cn/models/qwen/Qwen-Audio-Chat

点击复制model-id按钮

<img src="https://github.com/NascentCore/3k/assets/152252984/57d9aeae-45ee-41d1-bb5b-bca1e4550147" width="200">

model-id为`qwen/Qwen-Audio-Chat`

下载模型 `python3 3kctl.py download model modelscope qwen/Qwen-Audio-Chat`

正常情况输出：
```bash
model modelscope size 16 GB
PVC pvc-model-7b4ff4e80a400408 in namespace cpod created
Job 'download-model-7b4ff4e80a400408' created.
Custom Resource 'model-storage-7b4ff4e80a400408' created.
Custom Resource 'model-storage-7b4ff4e80a400408' status updated.
```

把上面信息保存到文档中，便于后面查看

查看download job状态
```bash
kubectl get job download-model-7b4ff4e80a400408 -n cpod
```

查看download pod状态
```bash
kubectl get pod -n cpod | grep download-model-7b4ff4e80a400408
```

- Running 下载中
- Completed 下载完成

查看模型文件
```yaml
# pod-pvc.yaml
apiVersion: v1
kind: Pod
metadata:
  name: alpine
  namespace: cpod
  labels:
    app: alpine
spec:
  volumes:
  - name: modelsave-pv
    persistentVolumeClaim:
      claimName: pvc-model-2e5ac6b350e54f96 # 替换成对应的pvc name
      readOnly: false
  containers:
  - name: alpine
    image: alpine
    stdin: true
    tty: true
    volumeMounts:
    - mountPath: "/data"
      name: modelsave-pv
```

```bash
kubectl apply -f pod-pvc.yaml
kubectl exec -it alpine -n cpod -- /bin/sh
```
进入到容器后
```yaml
ls -l /data
```

查看CRD状态
```bash
kubectl get ModelStorage model-storage-7b4ff4e80a400408 -n cpod -o yaml
```
