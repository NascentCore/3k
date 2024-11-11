import os
import sys
import shutil
import argparse
import requests
import oss2
from io import BytesIO
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from modelscope.hub.snapshot_download import snapshot_download

# 加载 .env 文件
load_dotenv()

# 阿里云 OSS 配置，从环境变量获取
OSS_CONFIG = {
    "endpoint": os.getenv("OSS_ENDPOINT"),
    "access_key_id": os.getenv("OSS_ACCESS_KEY_ID"),
    "access_key_secret": os.getenv("OSS_ACCESS_KEY_SECRET"),
    "bucket_name": os.getenv("OSS_BUCKET_NAME")
}

# API 配置
TASK_API = "https://llm.sxwl.ai/api/resource/task"
AUTHORIZATION_TOKEN = os.getenv("AUTHORIZATION_TOKEN")

# 计算目录大小
def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# 获取下载任务
def fetch_download_tasks():
    headers = {
        "Accept": "application/json",
        "Authorization": AUTHORIZATION_TOKEN
    }
    response = requests.get(TASK_API, headers=headers)
    response.raise_for_status()
    return response.json()["data"]

# 上报下载完成状态
def report_completion(task, error=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": AUTHORIZATION_TOKEN
    }
    data = {
        "data": [{
            "resource_id": task["resource_id"],
            "resource_type": task["resource_type"],
            "source": task["source"],
            "size": task["size"],
            "err": error or "",
            "ok": error is None
        }]
    }
    response = requests.put(TASK_API, headers=headers, json=data)
    response.raise_for_status()
    print(f"Reported completion for {task['resource_id']}")

# 检查 OSS 中是否已存在模型
def model_exists_in_oss(model_id):
    auth = oss2.Auth(OSS_CONFIG["access_key_id"], OSS_CONFIG["access_key_secret"])
    bucket = oss2.Bucket(auth, OSS_CONFIG["endpoint"], OSS_CONFIG["bucket_name"])
    # 检查 OSS 中是否有该模型文件夹（通过前缀匹配）
    return any(bucket.list_objects(prefix=model_id).object_list)

# 分片上传到阿里云 OSS
def multipart_upload_to_oss(bucket, oss_path, file_path):
    # 设置分片大小为 10 MB
    part_size = 10 * 1024 * 1024
    total_size = os.path.getsize(file_path)
    part_count = (total_size // part_size) + (1 if total_size % part_size else 0)
    
    upload_id = bucket.init_multipart_upload(oss_path).upload_id
    parts = []
    
    with open(file_path, 'rb') as f:
        for i in range(1, part_count + 1):
            part_data = f.read(part_size)
            part = bucket.upload_part(oss_path, upload_id, i, part_data)
            parts.append(oss2.models.PartInfo(i, part.etag))
    
    bucket.complete_multipart_upload(oss_path, upload_id, parts)
    print(f"{oss_path} uploaded successfully to OSS in parts.")

# 上传模型到阿里云 OSS，自动选择分片上传
def upload_to_oss(file_path, oss_path):
    auth = oss2.Auth(OSS_CONFIG["access_key_id"], OSS_CONFIG["access_key_secret"])
    bucket = oss2.Bucket(auth, OSS_CONFIG["endpoint"], OSS_CONFIG["bucket_name"])
    
    file_size = os.path.getsize(file_path)
    # 如果文件大小超过 10 MB，使用分片上传
    if file_size > 10 * 1024 * 1024:
        multipart_upload_to_oss(bucket, oss_path, file_path)
    else:
        with open(file_path, "rb") as f:
            bucket.put_object(oss_path, f)
        print(f"{oss_path} uploaded successfully to OSS.")

# 从 Hugging Face 下载模型
def download_and_upload_huggingface_model(model_id, token, cache_dir):
    model_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")

    api = HfApi()
    files = api.list_repo_files(model_id, token=token)
    for file in files:
        file_path = hf_hub_download(repo_id=model_id, filename=file, use_auth_token=token, cache_dir=cache_dir)
        oss_path = f"models/public/{model_id}/{file}"
        upload_to_oss(file_path, oss_path)

    total_size = get_directory_size(model_dir)
    shutil.rmtree(model_dir)  # 上传后删除文件，释放空间
    print(f"{model_dir} removed after upload.")
    return total_size

# 从 ModelScope 下载模型
def download_and_upload_modelscope_model(model_id, cache_dir):
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    total_size = get_directory_size(model_dir)

    for root, _, files in os.walk(model_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, model_dir)
            oss_path = f"models/public/{model_id}/{relative_path}"
            upload_to_oss(file_path, oss_path)

    shutil.rmtree(model_dir)  # 上传后删除文件，释放空间
    print(f"{file_path} removed after upload.")
    return total_size

# 主程序
def main(task):
    cache_dir = os.getenv("CACHE_DIR", "/data/cairong/cache")
    os.makedirs(cache_dir, exist_ok=True)

    if model_exists_in_oss(f"models/public/{task['resource_id']}"):
        print(f"Model {task['resource_id']} already exists in OSS. Skipping download.")
        return
    
    if task["resource_type"] != "model":
        print("Only support model download. Skipping download.")
        return

    if task["source"] == "huggingface":
        token = os.getenv("HF_TOKEN")
        size = download_and_upload_huggingface_model(task["resource_id"], token, cache_dir)
        task["size"] = size

    elif task["source"] == "modelscope":
        size = download_and_upload_modelscope_model(task["resource_id"], cache_dir)
        task["size"] = size

    else:
        print("Error: Unsupported model source. Use 'huggingface' or 'modelscope'.")
        return


if __name__ == "__main__":
    tasks = fetch_download_tasks()
    print(tasks)
    for task in tasks:
        print(task)
        try:
            main(task)
            print(task)
            report_completion(task)

        except Exception as e:
            print(f"Error processing task {task['resource_id']}: {e}")
            report_completion(task, error=str(e))