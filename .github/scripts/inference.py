import json
import time
import requests
import os 
import atexit
import sys
import signal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 配置stdout为无缓冲模式,使print立即输出
sys.stdout.reconfigure(line_buffering=True)

# 从环境变量获取基础URL，如果未设置则使用默认值
BASE_URL = os.getenv('SXWL_API_URL', 'https://llm.nascentcore.net')

# 全局变量用于存储需要清理的资源
resources_to_cleanup = {
    'inference_services': set(),
    'finetune_jobs': set()
}

def cleanup_handler(signum, frame):
    """处理中断信号,清理所有资源"""
    print("收到中断信号,开始清理资源...", flush=True)
    config = APIConfig.create_default()
    client = APIClient(config)
    
    # 清理推理服务
    for service_name in resources_to_cleanup['inference_services']:
        try:
            client.delete_inference_service(service_name)
        except Exception as e:
            print(f"清理推理服务 {service_name} 失败: {e}", flush=True)
    
    # 清理微调任务
    for job_id in resources_to_cleanup['finetune_jobs']:
        try:
            client.delete_finetune_job(job_id)
        except Exception as e:
            print(f"清理微调任务 {job_id} 失败: {e}", flush=True)
    
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)
signal.signal(signal.SIGQUIT, cleanup_handler)
signal.signal(signal.SIGHUP, cleanup_handler)

@dataclass
class APIConfig:
    base_url: str
    token: str
    headers: Dict[str, str]

    @classmethod
    def create_default(cls) -> 'APIConfig':
        token = os.environ["SXWL_TOKEN"]
        if token and isinstance(token, bytes):
            # 如果是 bytes，解码为字符串
            token = token.decode('utf-8')
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Origin': BASE_URL,
            'Connection': 'keep-alive',
            'User-Agent': 'github-actions'
        }
        return cls(BASE_URL, token, headers)

class APIClient:
    def __init__(self, config: APIConfig):
        self.config = config

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.config.base_url}/api{endpoint}"
        response = requests.request(method, url, headers=self.config.headers, **kwargs)
        response.raise_for_status()
        return response

    def get_models(self) -> List[Dict[str, Any]]:
        """获取可用的模型列表"""
        try:
            response = self._make_request('GET', '/resource/models')
            data = response.json()
            models = data.get('public_list', []) + data.get('user_list', [])
            print(f"获取到 {len(models)} 个模型", flush=True)
            return models
        except Exception as e:
            print(f"获取模型列表失败: {str(e)}", flush=True)
            return []

    def delete_inference_service(self, service_name: str) -> None:
        try:
            self._make_request('DELETE', '/job/inference', params={'service_name': service_name})
            print("推理服务删除成功", flush=True)
            resources_to_cleanup['inference_services'].discard(service_name)
        except Exception as e:
            print(f"删除推理服务失败: {str(e)}", flush=True)

    def delete_finetune_job(self, finetune_id: str) -> None:
        try:
            self._make_request('POST', '/userJob/job_del', json={'job_id': finetune_id})
            print("微调任务删除成功", flush=True)
            resources_to_cleanup['finetune_jobs'].discard(finetune_id)
        except Exception as e:
            print(f"删除微调任务失败: {str(e)}", flush=True)

class InferenceService:
    def __init__(self, client: APIClient):
        self.client = client
        self.service_name: Optional[str] = None
        self.api_endpoint: Optional[str] = None

    def deploy(self, model_config: Dict[str, Any]) -> None:
        response = self.client._make_request('POST', '/job/inference', json=model_config)
        self.service_name = response.json()['service_name']
        print(f"服务名称: {self.service_name}", flush=True)
        # 添加到需要清理的资源列表
        resources_to_cleanup['inference_services'].add(self.service_name)
        self._wait_for_ready()

    def _wait_for_ready(self, max_retries: int = 60, retry_interval: int = 30) -> None:
        for attempt in range(max_retries):
            response = self.client._make_request('GET', '/job/inference')
            status_json = response.json()
            
            for item in status_json.get('data', []):
                if item['service_name'] == self.service_name:
                    if item['status'] == 'running':
                        self.api_endpoint = item['api']
                        print(f"服务已就绪: {item}", flush=True)
                        return
                    break
            
            print(f"服务启动中... ({attempt + 1}/{max_retries})", flush=True)
            time.sleep(retry_interval)
        
        raise TimeoutError("服务启动超时")

    def chat(self, messages: list) -> Dict[str, Any]:
        if not self.api_endpoint:
            raise RuntimeError("服务尚未就绪")
            
        chat_url = self.api_endpoint
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        data = {"model": "/mnt/models", "messages": messages}
        
        response = requests.post(chat_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

class FinetuneJob:
    def __init__(self, client: APIClient):
        self.client = client
        self.job_id: Optional[str] = None
        self.adapter_id: Optional[str] = None

    def start(self, finetune_config: Dict[str, Any]) -> None:
        response = self.client._make_request('POST', '/job/finetune', json=finetune_config)
        self.job_id = response.json()['job_id']
        print(f"微调任务ID: {self.job_id}", flush=True)
        # 添加到需要清理的资源列表
        resources_to_cleanup['finetune_jobs'].add(self.job_id)
        self._wait_for_completion()
        self._get_adapter_id()

    def _wait_for_completion(self, max_retries: int = 60, retry_interval: int = 30) -> None:
        for _ in range(max_retries):
            print(f"正在检查微调任务状态... (第 {_ + 1}/{max_retries} 次尝试)", flush=True)
            response = self.client._make_request('GET', '/job/training', 
                                               params={'current': 1, 'size': 1000})
            
            print(f"API响应: {response.json()}", flush=True)
            for job in response.json().get('content', []):
                if job['jobName'] == self.job_id:
                    status = job['status']
                    print(f"微调状态: {status}", flush=True)
                    
                    if status == 'succeeded':
                        return
                    elif status in ['failed', 'error']:
                        raise RuntimeError("微调任务失败")
                    break
            
            time.sleep(retry_interval)
        raise TimeoutError("微调任务超时")

    def _get_adapter_id(self) -> None:
        response = self.client._make_request('GET', '/resource/adapters')
        
        for adapter in response.json().get('user_list', []):
            try:
                meta = json.loads(adapter.get('meta', '{}'))
                if meta.get('finetune_id') == self.job_id:
                    self.adapter_id = adapter['id']
                    print(f"适配器ID: {self.adapter_id}", flush=True)
                    return
            except json.JSONDecodeError:
                continue
        
        raise ValueError(f"未找到对应的适配器")

def main():
    try:
        # 初始化API客户端
        config = APIConfig.create_default()
        client = APIClient(config)
        
        # 获取可用模型列表
        available_models = client.get_models()
        # 检查目标模型是否在可用模型列表中
        target_model_id = "model-storage-0ce92f029254ff34"
        model_found = False
        for model in available_models:
            if model.get('id') == target_model_id:
                model_found = True
                print(f"找到目标模型: {model.get('name')}", flush=True)
                break

        if not model_found:
            raise ValueError(f"未找到ID为 {target_model_id} 的模型")
        
        # 部署基础模型推理服务
        base_model = InferenceService(client)
        base_model_config = {
            "gpu_model": "NVIDIA-GeForce-RTX-3090",
            "model_category": "chat",
            "gpu_count": 1,
            "model_id": model.get('id'),
            "model_name":model.get('name'),
            "model_size": 15065904829,
            "model_is_public": True,
            "model_template": "gemma",
            "min_instances": 1,
            "model_meta": "{\"template\":\"gemma\",\"category\":\"chat\",\"can_finetune\":true,\"can_inference\":true}",
            "max_instances": 1
        }
        base_model.deploy(base_model_config)
        
        # 测试基础模型对话
        response = base_model.chat([{"role": "user", "content": "你是谁"}])
        print("基础模型响应:", response, flush=True)
        
        # 开始微调任务
        finetune = FinetuneJob(client)
        finetune_config = {
            "model": model.get('name'),
            "training_file": "dataset-storage-f90b82cc7ab88911",
            "gpu_model": "NVIDIA-GeForce-RTX-3090",
            "gpu_count": 1,
            "finetune_type": "lora",
            "hyperparameters": {
                "n_epochs": "3.0",
                "batch_size": "4",
                "learning_rate_multiplier": "5e-5"
            },
            "model_saved_type": "lora",
            "model_id": model.get('id'),
            "model_name": model.get('name'),
            "model_size": model.get('size'),
            "model_is_public": model.get('is_public'),
            "model_template": model.get('template'),
            "model_category": "chat",
            "dataset_id": "dataset-storage-f90b82cc7ab88911",
            "model_meta": model.get('meta'),
            "dataset_name": "llama-factory/alpaca_data_zh_short",
            "dataset_size": 14119,
            "dataset_is_public": True
        }
        finetune.start(finetune_config)
        
        # 部署微调后的模型
        finetuned_model = InferenceService(client)
        finetuned_model_config = {**base_model_config, 
                                "adapter_id": finetune.adapter_id,
                                "adapter_is_public": False}
        finetuned_model.deploy(finetuned_model_config)
        
        # 测试微调后的模型对话
        response = finetuned_model.chat([{"role": "user", "content": "你是谁"}])
        print("微调后模型响应:", response, flush=True)
    except Exception as e:
        print(f"发生错误: {e}", flush=True)
        cleanup_handler(None, None)
        raise

if __name__ == "__main__":
    main()
