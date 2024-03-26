import os
import math
import subprocess
import time
import hashlib
from kubernetes import client, config

def get_hashed_name(name, length=16):
    return hashlib.sha1(name.encode()).hexdigest()[:length]

# 检查PVC是否存在
def check_pvc_exists(namespace, pvc_name):
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    pvcs = api_instance.list_namespaced_persistent_volume_claim(namespace=namespace)
    for pvc in pvcs.items:
        if pvc.metadata.name == pvc_name:
            return True
    return False

# 检查CRD是否存在
def check_crd_exists(namespace, crd_name, api_version, plural):
    config.load_kube_config()
    api_instance = client.CustomObjectsApi()
    try:
        api_instance.get_namespaced_custom_object(group=api_version.split('/')[0], version=api_version.split('/')[1], namespace=namespace, plural=plural, name=crd_name)
        return True
    except client.rest.ApiException as e:
        if e.status != 404:
            print(f"Error checking CRD: {e}")
        return False

# 计算目录大小
def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

# 创建PVC
def create_pvc(namespace, name, size_in_bytes):
    size_in_gib = max(1, math.ceil(size_in_bytes / (1024**3)))
    config.load_kube_config()
    pvc = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteMany"],
            resources=client.V1ResourceRequirements(
                requests={"storage": f"{size_in_gib}Gi"}
            )
        )
    )
    api_instance = client.CoreV1Api()
    api_instance.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
    print(f"PVC {name} created in namespace {namespace}, size {size_in_gib} Gi")

# 创建Pod并挂载PVC
def create_pod_with_pvc(namespace, pod_name, pvc_name):
    config.load_kube_config()
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=pod_name),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="busybox",
                    image="busybox",
                    volume_mounts=[client.V1VolumeMount(mount_path="/workspace", name="storage")],
                    command=["sleep", "3600"]
                )
            ],
            volumes=[
                client.V1Volume(
                    name="storage",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)
                )
            ]
        )
    )
    api_instance = client.CoreV1Api()
    api_instance.create_namespaced_pod(namespace=namespace, body=pod)
    print(f"Pod {pod_name} created in namespace {namespace} with PVC {pvc_name}")

# 复制目录内容到PVC
def copy_to_pvc(namespace, pod_name, src_dir, target_dir='/workspace'):
    # 列出目录下的所有文件和文件夹，然后逐一复制
    for item in os.listdir(src_dir):
        full_path = os.path.join(src_dir, item)
        if os.path.isdir(full_path):
            target = f"{namespace}/{pod_name}:{target_dir}/{item}"
        else:
            target = f"{namespace}/{pod_name}:{target_dir}"
        cmd = f"kubectl cp {full_path} {target}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"Content from {full_path} copied to PVC mounted on {pod_name}")

# 创建CRD对象
def create_crd(namespace, data_type, name, crd_name, pvc_name, api_version):
    config.load_kube_config()

    if data_type == 'model':
        kind = 'ModelStorage'
        spec = {
            'modeltype': 'local',
            'modelname': name,
            'pvc': pvc_name
        }
    elif data_type == 'dataset':
        kind = 'DataSetStorage'
        spec = {
            'datasettype': 'local',
            'datasetname': name,
            'pvc': pvc_name
        }
    else:
        raise ValueError("Invalid data type. Must be 'model' or 'dataset'.")

    crd_body = {
        'apiVersion': api_version,
        'kind': kind,
        'metadata': {
            'name': crd_name,
            'namespace': namespace
        },
        'spec': spec
    }

    api_instance = client.CustomObjectsApi()
    api_instance.create_namespaced_custom_object(
        group=api_version.split('/')[0],
        version=api_version.split('/')[1],
        namespace=namespace,
        plural=f"{data_type}storages",
        body=crd_body
    )
    print(f"{kind} {crd_name} created in namespace {namespace}")

# 删除Pod
def delete_pod(namespace, pod_name):
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    api_instance.delete_namespaced_pod(name=pod_name, namespace=namespace)
    print(f"Pod {pod_name} deleted in namespace {namespace}")

# 检查Pod状态是否为Running
def wait_for_pod_ready(namespace, pod_name, timeout=300):
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    start_time = time.time()

    while time.time() - start_time < timeout:
        pod = api_instance.read_namespaced_pod(namespace=namespace, name=pod_name)
        if pod.status.phase == "Running":
            print(f"Pod {pod_name} is now in Running state")
            return
        else:
            print(f"Waiting for pod {pod_name} to be in Running state...")
            time.sleep(10)

    raise Exception(f"Pod {pod_name} is not in Running state after {timeout} seconds")

# 更新CRD状态
def update_crd_status(namespace, crd_name, data_type, api_version, dir_size, status):
    config.load_kube_config()
    api_instance = client.CustomObjectsApi()
    plural = f"{data_type}storages"

    # 构建状态更新体
    status_update = {
        "apiVersion": api_version,
        "kind": "DataSetStorage" if data_type == 'dataset' else "ModelStorage",
        "metadata": {
            "name": crd_name,
            "namespace": namespace
        },
        "status": {
            "size": dir_size,
            "phase": status
        }
    }

    # 更新CRD状态
    api_instance.patch_namespaced_custom_object_status(
        group=api_version.split('/')[0],
        version=api_version.split('/')[1],
        namespace=namespace,
        plural=plural,
        name=crd_name,
        body=status_update
    )
    print(f"CRD {crd_name} status updated to {status}")

# 删除CRD和PVC
def delete_crd_and_pvc(namespace, resource_type, resource_name, api_version, plural):
    config.load_kube_config()
    api_instance = client.CustomObjectsApi()
    core_v1_api = client.CoreV1Api()

    # 删除CRD
    if resource_type == "crd":
        try:
            api_instance.delete_namespaced_custom_object(
                group=api_version.split('/')[0],
                version=api_version.split('/')[1],
                namespace=namespace,
                plural=plural,
                name=resource_name
            )
            print(f"CRD {resource_name} deleted")
        except client.exceptions.ApiException as e:
            print(f"No existing CRD {resource_name} to delete")

    # 删除PVC
    if resource_type == "pvc":
        try:
            core_v1_api.delete_namespaced_persistent_volume_claim(
                namespace=namespace,
                name=resource_name
            )
            print(f"PVC {resource_name} deleted")
        except client.exceptions.ApiException as e:
            print(f"No existing PVC {resource_name} to delete")
