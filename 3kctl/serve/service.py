from datetime import datetime
import signal
from flask import Flask, jsonify, request
from plumbum import cli
import subprocess
import yaml
import json
import os
import paramiko
import threading
import requests
from requests.auth import HTTPBasicAuth
from cli.deploy.utils import parse_ini

app = Flask(__name__)

conf = parse_ini()

@app.route('/nodes', methods=['GET'])
def get_nodes():
    try:
        # 调用kubectl命令获取节点信息
        result = subprocess.run(['kubectl', 'get', 'nodes', '-o', 'json'], capture_output=True, text=True)

        # 检查命令是否成功执行
        if result.returncode != 0:
            return jsonify({'error': 'Failed to get nodes'}), 500
        
        # 解析JSON数据并返回
        nodes = []
        for item in json.loads(result.stdout)['items']:
            node = {'role': []}
            node['name'] = item['metadata']['name']
            node['capacity'] = item['status']['capacity']
            if 'node-role.kubernetes.io/control-plane' in item['metadata']['labels']:
                node['role'].append('control-plane')
            if 'node-role.kubernetes.io/worker' in item['metadata']['labels']:
                node['role'].append('worker')
            if 'nvidia.com/gpu.product' in item['metadata']['labels']:
                node['gpu_product'] = item['metadata']['labels']['nvidia.com/gpu.product']
            node['gpu_count'] = int(item['status']['capacity']['nvidia.com/gpu'])
            nodes.append(node)

        return jsonify(nodes)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-node', methods=['POST'])
def add_node():
    try:
        # 获取前端传来的数据
        params = request.get_json()

        if not test_ssh(params):
            return jsonify({'error': 'The SSH service is not available.'}), 500
        
        role = params.get('node_role')
        new_host = {
            'name': params.get('node_name'),
            'internalAddress': params.get('node_ip'),
            'port': params.get('ssh_port'),
            'user': params.get('ssh_user'),
            'password': params.get('ssh_password')
        }

        # 读取YAML文件内容
        with open('conf/config-sample.yaml', 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                return jsonify({'error': str(exc)}), 500
            
        # 检查是否存在指定的host
        host_exists = False
        for host in data['spec']['hosts']:
            if host['name'] == new_host['name']:
                host_exists = True
                break

        # 如果不存在，则添加新的host
        if not host_exists:
            data['spec']['hosts'].append(new_host)

        if new_host['name'] not in data['spec']['roleGroups'][role]:
            data['spec']['roleGroups'][role].append(new_host['name'])

        # 将更新后的数据写回文件
        with open('conf/config-sample.yaml', 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        print(f"Host '{new_host['name']}' has been added to the YAML file.")

        def on_complete(result):
        # 这里处理异步任务完成后的逻辑
            print(result)
        
        # 调用3kctl命令执行添加节点操作
        add_node_command = f'sudo 3kctl deploy add nodes'
        async_subprocess_run(add_node_command, callback=on_complete)

        # 立即返回响应，不等待异步任务完成
        return jsonify({'message': 'Node addition initiated'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/build_image', methods=['POST'])
def build_image():
    try:
        data = request.get_json()
        base_image = data.get('base_image')
        user_id = data.get('user_id')
        instance_name = data.get('instance_name')

        # 创建线程来异步执行镜像构建和推送
        thread = threading.Thread(target=build_and_push_image, args=(base_image, user_id, instance_name))
        thread.start()

        return jsonify({'message': 'Image build and push initiated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/repos', methods=['GET'])
def get_repos():
    try:
        project_name = request.args.get('project_name')
        if not project_name:
            return jsonify({"error": "project_name is required"}), 400
        
        repository = request.args.get('repository')
        if repository:
            res = list_tags(project_name, repository)
        else:
            res = list_repositories(project_name)

        return jsonify(res)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class Serve(cli.Application):
    """启动和停止 Flask 服务"""

    PID_FILE = './gunicorn.pid'
    action_performed = False

    @cli.switch(["start"])
    def start(self):
        self.action_performed = True
        print("Starting Flask server with Gunicorn...")
        subprocess.run([
            '/opt/.venv/bin/gunicorn',
            '-w', '4',
            '-b', '0.0.0.0:5000',
            '--daemon',
            '--pid', self.PID_FILE,
            'cli.serve.service:app'
        ])
        print("Flask server is running in the background.")

    @cli.switch(["stop"])
    def stop(self):
        self.action_performed = True
        print("Stopping Flask server...")
        if os.path.exists(self.PID_FILE):
            with open(self.PID_FILE, 'r') as file:
                pid = int(file.read())
            os.kill(pid, signal.SIGTERM)
            print("Flask server stopped.")
        else:
            print("PID file not found. Is the Flask server running?")

    def main(self):
        if not self.action_performed:
            print("No command given. Available commands are --start and --stop")

def test_ssh(data):
    """
    测试 SSH 服务是否可达。

    :param host: 要连接的主机名或IP地址
    :param port: SSH 端口，默认是 22
    :param username: SSH 用户名
    :param password: SSH 密码
    :return: 布尔值，表示是否成功连接
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(data["node_ip"], port=data["ssh_port"], username=data["ssh_user"], password=data["ssh_password"], timeout=3)
        # 如果连接成功，返回 True
        return True
    except Exception as e:
        print(f"SSH 连接失败: {e}")
        # 如果连接失败，返回 False
        return False
    finally:
        client.close()

def async_subprocess_run(command, callback=None):
    """
    在后台线程中异步执行命令，并可选地在完成时调用回调函数。

    :param command: 要执行的命令，字符串形式。
    :param callback: 命令完成时调用的回调函数，应该接受一个参数：命令的输出结果。
    """
    def run_in_thread(command, callback):
        # 使用 subprocess.Popen 而不是 run 来启动子进程
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()  # 等待命令执行完成并获取输出
        
        # 如果提供了回调函数，则调用它
        if callback:
            callback({'stdout': stdout, 'stderr': stderr, 'returncode': process.returncode})

    # 创建并启动后台线程来执行命令
    thread = threading.Thread(target=run_in_thread, args=(command, callback))
    thread.start()

def build_and_push_image(base_image, user_id, instance_name):
    workdir = f'/tmp/{user_id}/{instance_name}'
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # 复制代码到工作目录
    code_dir = f'jupyterlab-{user_id}-{instance_name}:/workspace'
    copy_cmd = ['kubectl', 'cp', '-n', 'cpod', code_dir, './workspace']
    copy_cmd = ['kubectl', 'cp', code_dir, './workspace']
    subprocess.run(copy_cmd, check=True)

    # 生成Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(f'FROM {base_image}\n')
        f.write('WORKDIR /workspace\n')
        f.write(f'COPY ./workspace /workspace\n')
        f.write('RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple\n')
    # 构建完整的镜像标签
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    full_image_name = f'dockerhub.kubekey.local/{user_id}/{instance_name}:v{timestamp}'

    # 创建项目
    create_project(user_id)

    # 构建Docker镜像
    build_cmd = ['sudo', 'docker', 'build', '-t', full_image_name, '.']
    subprocess.run(build_cmd, check=True)

    # 推送Docker镜像
    push_cmd = ['sudo', 'docker', 'push', full_image_name]
    subprocess.run(push_cmd, check=True)

    return full_image_name

def create_project(project_name, public=True):
    url = "https://dockerhub.kubekey.local/api/v2.0/projects"
    data = {
        "project_name": project_name,
        "public": public
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=data, headers=headers, auth=HTTPBasicAuth(conf.registry.harbor_user, conf.registry.harbor_pass), verify=False)
    
    if response.status_code == 201:
        print(f"Project {project_name} created successfully.")
    else:
        print(f"Failed to create project {project_name}. Status code: {response.status_code}, Response: {response.text}")

def list_repositories(project_name):
    hub = "dockerhub.kubekey.local"
    url = f"https://{hub}/api/v2.0/projects/{project_name}/repositories?page=1&page_size=100"
    response = requests.get(url, auth=HTTPBasicAuth(conf.registry.harbor_user, conf.registry.harbor_pass), verify=False)
    if response.status_code == 200:
        res = response.json()
        repos = []
        for i in res:
            repos.append(f"{hub}/{i['name']}")
        return repos
    else:
        print(f"Error fetching repositories. Status code: {response.status_code}")
        return None

def list_tags(project_name, repository):
    hub = "dockerhub.kubekey.local"
    url = f"https://{hub}/api/v2.0/projects/{project_name}/repositories/{repository}/artifacts?page=1&page_size=100"
    response = requests.get(url, auth=HTTPBasicAuth(conf.registry.harbor_user, conf.registry.harbor_pass), verify=False)
    if response.status_code == 200:
        res = response.json()
        tags = []
        for i in res:
            if i['tags']:
                tags.append(f"{hub}/{project_name}/{repository}:{i['tags'][0]['name']}")
        return tags
    else:
        print(f"Error fetching tags. Status code: {response.status_code}")
        return None