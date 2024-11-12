import time
import requests
import os 


# 部署推理
url = 'https://llm.sxwl.ai/api/job/inference'
headers = {
'Accept': 'application/json, text/plain, */*',
'Authorization': f'Bearer {os.environ["SXWL_TOKEN"]}',
'Connection': 'keep-alive',
'Content-Type': 'application/json',
'Origin': 'https://llm.sxwl.ai',
'Referer': 'https://llm.sxwl.ai/models',
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
}
data = {
"gpu_model": "NVIDIA-GeForce-RTX-3090",
"model_category": "chat",
"gpu_count": 1,
"model_id": "model-storage-0ce92f029254ff34",
"model_name": "google/gemma-2b-it",
"model_size": 15065904829,
"model_is_public": True,
"model_template": "gemma",
"min_instances": 1,
"max_instances": 1
}

response = requests.post(url, headers=headers, json=data)

if response.status_code != 200:
    print(f"请求失败，状态码: {response.status_code}")
    print(f"错误信息: {response.text}")
    exit(1)

response_json = response.json()
service_name = response_json['service_name']
print(f"服务名称: {service_name}")

# 获取推理状态及地址

# 获取推理状态
status_url = 'https://llm.sxwl.ai/api/job/inference'
status_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh,en;q=0.9,zh-CN;q=0.8,pt;q=0.7',
    'Authorization': headers['Authorization'],
    'Connection': 'keep-alive',
    'Referer': 'https://llm.sxwl.ai/jobdetail',
    'User-Agent': headers['User-Agent'],
}

status_response = requests.get(status_url, headers=status_headers)

if status_response.status_code != 200:
    print(f"获取状态失败，状态码: {status_response.status_code}")
    print(f"错误信息: {status_response.text}")
    exit(1)

status_json = status_response.json()
# 解析状态响应
max_retries = 6
retry_count = 0
while retry_count < max_retries:
    if 'data' in status_json:
        # 遍历所有数据查找匹配的service_name
        inference_info = None
        for item in status_json['data']:
            if item['service_name'] == service_name:
                inference_info = item
                break
        
        if inference_info:
            status = inference_info['status']
            model_name = inference_info['model_name']
            url = inference_info['url']
            api = inference_info['api']
            create_time = inference_info['create_time']
            
            print(f"服务名称: {service_name}")
            print(f"状态: {status}")
            print(f"模型名称: {model_name}")
            print(f"服务URL: {url}")
            print(f"API地址: {api}")
            print(f"创建时间: {create_time}")

            if status == 'ready':
                break
        else:
            print(f"未找到服务名称为 {service_name} 的推理任务")
            exit(1)
    else:
        print("未找到任何推理任务信息")
        exit(1)

    retry_count += 1
    if retry_count < max_retries:
        print(f"等待10秒后重试... ({retry_count}/{max_retries})")
        time.sleep(10)
        status_response = requests.get(status_url, headers=status_headers)
        if status_response.status_code == 200:
            status_json = status_response.json()
        else:
            print(f"获取状态失败，状态码: {status_response.status_code}")
            print(f"错误信息: {status_response.text}")
            exit(1)

if retry_count >= max_retries and status != 'ready':
    print("推理服务启动超时")
    exit(1)



# 调用推理对话请求

# 删除推理服务

# 删除推理服务
print("开始删除推理服务...")
delete_url = f"https://llm.sxwl.ai/api/job/inference"
delete_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Authorization': 'Bearer eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkOWUxYWNlMTliM2I0ZmNmOGZjZTgyYzBkMWYyZjMwMiIsInN1YiI6ImRnQHN4d2wuYWkiLCJ1c2VyX2lkIjoidXNlci01NGFlNjZhMi1lMDUzLTQwNmQtYTBiNC05OTE5YzQ3YjNhZjYiLCJ1c2VyaWQiOjE2NSwidXNlcm5hbWUiOiJkZ0BzeHdsLmFpIn0.s057vqqy9oWhkwojMkPFkIprB82s1EcUOfghrPCiERPjNSjR90GWCYijs9-pb73JnYZaoOCYaPDTDu2xoy3Udg',
    'Origin': 'https://llm.sxwl.ai',
    'Referer': 'https://llm.sxwl.ai/jobdetail'
}

delete_params = {
    'service_name': service_name
}

delete_response = requests.delete(delete_url, headers=delete_headers, params=delete_params)

if delete_response.status_code == 200:
    print("推理服务删除成功")
else:
    print(f"删除推理服务失败，状态码: {delete_response.status_code}")
    print(f"错误信息: {delete_response.text}")
    exit(1)




# 微调

# 获取微调状态

# 删除微调

# 使用微调之后的适配器部署推理

# 获取适配器推理状态及地址

# 调用适配器推理对话请求

# 删除适配器推理服务
