# 3kctl
3kctl是3k平台的命令行工具

## 3kctl deploy

## 3kctl download
### 测试用例
在魔搭上找到自己需要的模型，例如https://modelscope.cn/models/qwen/Qwen-Audio-Chat

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
正常应该显示 Running

查看CRD状态