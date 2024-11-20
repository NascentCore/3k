import os
import sys
import json
import requests
import time

class GPUJobTester:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'User-Agent': 'Python-Test-Script'
        }

    def submit_job(self, job_config):
        """提交GPU任务"""
        url = f"{self.base_url}/api/job/training"
        response = requests.post(url, headers=self.headers, json=job_config)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("job_id")
        print(f"任务提交成功, 任务ID: {job_id}")
        return job_id

    def check_job_status(self, job_id, max_retries=60, interval=30):
        """检查任务状态"""
        url = f"{self.base_url}/api/job/training"
        for attempt in range(max_retries):
            print(f"检查任务状态 (尝试 {attempt + 1}/{max_retries})...")
            response = requests.get(url, headers=self.headers, params={"current": 1, "size": 1000})
            response.raise_for_status()
            jobs = response.json().get('content', [])
            for job in jobs:
                if job['jobName'] == job_id:
                    status = job['status']
                    print(f"任务状态: {status}")
                    if status in ['succeeded', 'failed', 'error']:
                        return status
            time.sleep(interval)
        raise TimeoutError("检查任务状态超时")

    def delete_job(self, job_id):
        """删除指定任务"""
        url = f"{self.base_url}/api/userJob/job_del"
        payload = {"job_id": job_id}
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            print(f"任务 {job_id} 删除成功")
        else:
            print(f"删除任务 {job_id} 失败, 错误: {response.text}")

def main():
    # 配置环境变量
    BASE_URL = os.getenv('SXWL_API_URL', 'https://llm.sxwl.ai')
    TOKEN = os.getenv('SXWL_TOKEN', 'your-token-here')

    # 初始化测试器
    tester = GPUJobTester(BASE_URL, TOKEN)

    # 定义任务配置
    job_config = {
        "ckptPath": "/workspace/ckpt",
        "ckptVol": 100,
        "modelPath": "/workspace/saved_model",
        "modelVol": 100,
        "nodeCount": 1,
        "gpuNumber": 1,
        "gpuType": "NVIDIA-GeForce-RTX-3090",
        "imagePath": "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/generaljob-example:v2",
        "jobType": "General",
        "dataset_id": "dataset-storage-5172f0de7b73c55a",
        "dataset_path": "/mnt/datasets",
        "dataset_name": "llama-factory/oaast_sft_zh",
        "dataset_size": 1069203,
        "dataset_is_public": True
    }

    try:
        # 提交任务
        job_id = tester.submit_job(job_config)

        # 检查任务状态
        status = tester.check_job_status(job_id)
        print(f"任务最终状态: {status}")

        # 如果任务失败或出错，则删除任务
        if status in ['failed', 'error']:
            print("任务状态异常, 开始删除任务...")
            tester.delete_job(job_id)

    except Exception as e:
        print(f"任务处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
