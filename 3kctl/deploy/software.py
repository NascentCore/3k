import subprocess
import logging
import time
import os
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATUS_FILE = 'install_status.json'

def load_installed_softwares():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as file:
            return set(json.load(file))
    return set()


def update_installed_softwares(installed_softwares):
    with open(STATUS_FILE, 'w') as file:
        json.dump(list(installed_softwares), file)


def install_with_helm(name, namespace, values):
    try:
        cmd = ["helm", "install", "--namespace", namespace, "--create-namespace", name, f"harbor/{name}"]
        if values:
            cmd.extend(["-f", f"deploy/values/{values}"])
        subprocess.run(cmd, check=True)
        logger.info(f"Helm 安装 {name} 成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"Helm 安装 {name} 失败: {e}")
        raise


def apply_with_kubectl(file):
    try:
        subprocess.run(["kubectl", "apply", "-f", file], check=True)
        logger.info(f"Kubectl 应用 {file} 成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"Kubectl 应用 {file} 失败: {e}")
        raise


def get_pod_status(app_name, namespace):
    try:
        result = subprocess.run(["kubectl-assert", "pod-ready", "-n", namespace], capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"获取 {app_name} 的 Pod 状态失败: {e}")
        return None


def check_pod_status(app_name, namespace, timeout=300, interval=10):
    """
    循环检查 Pod 的状态，直到其变为 'Running' 或超时。

    :param app_name: 应用的名称
    :param timeout: 超时时间（秒）
    :param interval: 检查间隔（秒）
    :return: 如果 Pod 成功运行则返回 True，否则返回 False
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_pod_status(app_name, namespace)
        if status is None:
            break  # 获取状态失败，跳出循环
        if status.returncode == 0 and "No resources found in network namespace" not in status.stdout:
            logger.info(f"{app_name} 的 Pod 状态为 Running")
            return True
        else:
            logger.info(f"{app_name} 的 Pod 正在启动，等待中...")
            time.sleep(interval)

    logger.error(f"检查 {app_name} 的 Pod 状态超时")
    return False


def install_software(software, installed_softwares, softwares):
    """
    安装软件并根据需要检查依赖软件的状态。

    :param software: 要安装的软件字典
    :param installed_softwares: 已安装软件的集合，用于避免重复安装
    """
    name = software["name"]

    # 检查软件是否已经安装
    if name in installed_softwares or not software["deploy"]:
        logger.info(f"{name} 已经安装或无需安装，跳过")
        return

    # 安装依赖
    for dep_name in software.get("dependencies", []):
        dep_software = find_software_by_name(dep_name, softwares)
        install_software(dep_software, installed_softwares, softwares)

    # 安装软件本身
    try:
        if software["type"] == "helm":
            if name == "rook-ceph-cluster":
                add_node_label(software["deployNode"])
                
            install_with_helm(name, software["namespace"], software["values"])
        elif software["type"] == "kubectl":
            apply_with_kubectl(software["file"])

        # 检查软件的 Pod 是否成功运行
        if not check_pod_status(name, software["namespace"]):
            raise Exception(f"{name} 未能成功运行")

        # 添加到已安装软件集合
        installed_softwares.add(name)
        logger.info(f"{name} 安装并运行成功")

    except Exception as e:
        logger.error(f"安装 {name} 失败: {e}")
        raise


def find_software_by_name(name, softwares):
    for software in softwares:
        if software["name"] == name:
            return software
    raise ValueError(f"软件 {name} 未定义")


def add_node_label(nodes):
    for node in nodes:
        try:
            subprocess.run(["kubectl", "label", "node", node, "role=ceph", "--overwrite"], check=True)
            logger.info(f"Kubectl 增加 {node} 节点标签 role=ceph 成功")
        except subprocess.CalledProcessError as e:
            logger.error(f"Kubectl 增加 {node} 节点标签 role=ceph 失败: {e}")
            raise