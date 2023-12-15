# 3kctl
3kctl是3k平台的命令行工具

## 3kctl deploy

## 3kctl download
download子命令用于下载模型、数据集。
### 查看下载任务状态
```bash
python3 3kctl.py download status
```
输出:

<img width="800" alt="image" src="https://github.com/NascentCore/3k/assets/152252984/0b10010c-93f9-40c9-8f48-666f14f163c8">

- HUB: 模型市场
- MODEL_ID: 模型ID
- HASH: 模型的唯一标识
- PHASE: 当前下载状态

### 下载模型
查看命令download model的参数
```bash
python3 3kctl.py download model -h
```
```bash
3kctl download model 0.1

download model

Usage:
    3kctl download model [SWITCHES] hub_name model_id [proxy=] [depth=1] [downloader_version=v0.0.4] [namespace=cpod]

Meta-switches:
    -h, --help         Prints this help message and quits
    --help-all         Prints help messages of all sub-commands and quits
    -v, --version      Prints the program's version and quits

```
- proxy 设置vpn，格式`http://server-ip:port/`。默认不走代理
- depth 设置git clone --depth，默认1，也就是最近一次提交的内容，没有历史记录，这样可以节省空间
- downloader_version 下载器版本，默认为最新版

#### 魔搭下载示例
在魔搭上找到自己需要的模型，例如 [https://modelscope.cn/models/qwen/Qwen-Audio-Chat](https://modelscope.cn/models/qwen/Qwen-Audio-Chat)

点击复制model-id按钮

<img width="300" src="https://github.com/NascentCore/3k/assets/152252984/57d9aeae-45ee-41d1-bb5b-bca1e4550147">

model-id为`qwen/Qwen-Audio-Chat`

下载模型 `python3 3kctl.py download model modelscope qwen/Qwen-Audio-Chat`

#### HuggingFace下载示例
在HuggingFace上找到自己需要的模型，例如 [https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)

点击复制model-id按钮

model-id为`IDEA-CCNL/Ziya-LLaMA-13B-v1`

下载模型 `python3 3kctl.py download model huggingface IDEA-CCNL/Ziya-LLaMA-13B-v1 http://127.0.0.1:6789`


#### 下载命令输出
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
