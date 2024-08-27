import os
import json
import requests
import argparse

# ANSI转义码前缀
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# 前景色
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

def get_knowledge_list():
    url = f"{base_url}/api/local_doc_qa/list_knowledge_base"
    data = {
        "user_id": "zzp"
    }

    response = requests.post(url, data=data)
    resp = response.json()
    return resp["data"]

def find_kb_id_by_name(name):
    data = get_knowledge_list()
    return next((item['kb_id'] for item in data if item['kb_name'] == name), None)

def delete_knowledge(kb_id):
    url = f"{base_url}/api/local_doc_qa/delete_knowledge_base"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "user_id": "zzp",
        "kb_ids": [kb_id]
    }

    response = requests.post(url, headers=headers,  data=json.dumps(data))
    resp = response.json()
    if resp["code"] == 200:
        print(f"{GREEN}已删除该知识库，kb_id: {kb_id}{RESET}")

def create_knowledge(name):
    url = f"{base_url}/api/local_doc_qa/new_knowledge_base"
    data = {
        "user_id": "zzp",
        "kb_name": name
    }
    response = requests.post(url, data=data)
    if response.headers.get('Content-Type') == 'application/json':
        resp = response.json()
    else:
        print("Response is not JSON format:", response.text)
        return None
    return resp["data"]["kb_id"]

def get_file_list(kb_id):
    url = f"{base_url}/api/local_doc_qa/list_files"
    data = {
        "user_id": "zzp",
        "kb_id": kb_id
    }

    response = requests.post(url, data=data)
    resp = response.json()
    return resp["data"]

def check_file_status(kb_id, file_name):
    data = get_file_list(kb_id)
    for detail in data['details']:
        if detail['file_name'] == file_name and detail['status'] in ['green', 'gray']:
            return detail['file_id']
    return False

def upload_files(kb_id, folder_path):
    url = f"{base_url}/api/local_doc_qa/upload_files"
    data = {
        "user_id": "zzp",
        "kb_id": kb_id,
        "mode": "strong",
        "use_local_file": "false"
    }

    files = []
    extensions = ['.md', '.txt', '.pdf', '.jpg', 'png', '.docx', '.xlsx',
                  'pptx', '.eml', '.csv']
    for root, dirs, file_names in os.walk(folder_path):
        for file_name in file_names:
            if file_name.endswith(tuple(extensions)):
                file_path = os.path.join(root, file_name)
                file_id = check_file_status(kb_id, file_name)
                if file_id:
                    print(f"{YELLOW}{file_name} 已存在，file_id: {file_id}{RESET}")
                    continue
                files.append(("files", open(file_path, "rb")))
    if len(files) > 0:
        response = requests.post(url, files=files, data=data)
        resp = response.json()
        for i in resp["data"]:
            print(f"{GREEN}{i['file_name']} 已上传，file_id: {i['file_id']}{RESET}")

def main():
    directory = args.datadir
    names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    for name in names:
        print(f"===== 知识库：{name} =====")
        kb_id = find_kb_id_by_name(name)
        if kb_id:
            print(f"{YELLOW}该知识库已存在，kb_id: {kb_id}{RESET}")
            if args.clean:
                delete_knowledge(kb_id)
                kb_id = None
        if not kb_id:
            kb_id = create_knowledge(name)
            if not kb_id:
                continue
            else:
                print(f"{GREEN}已创建名为『{name}』的知识库，kb_id: {kb_id}{RESET}")

        upload_files(kb_id, os.path.join(directory, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建知识库以及批量上传文档")
    parser.add_argument("-d", "--datadir", default="./data", help="指定上传导入的知识库及文档所在目录")
    parser.add_argument("-c", "--clean", action="store_true", help="知识库存在的情况下，删除该知识库并重新创建并导入文档")

    args = parser.parse_args()
    base_url = "http://qanything.llm.sxwl.ai:30003"

    main()
