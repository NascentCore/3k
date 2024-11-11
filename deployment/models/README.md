# 同步模型
`sync_model.py` 用于从 huggingface 或 modelscpoe 同步开源模型到算想云。

## 功能
- 从算想云 API 获取同步模型列表
- 从 huggingface 或 modelscope 下载模型
- 同步模型至算想云 OSS
- 更新算想云模型状态

## 部署
- 在新加坡节点部署该脚本
```crontab
* * * * * cd /data/cairong && /home/cairong/.venv/bin/python sync_model.py
```