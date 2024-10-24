#!/bin/bash
echo 'app start!';

API_PORT=8080 python /app/src/api.py --model_name_or_path /mnt/models --infer_backend vllm --template llama3  --vllm_maxlen=8192 &

# 等待 vllm 启动完成
sleep 20

API_URL=http://localhost:8080/v1/chat/completions python3 /app/web/api/app.py