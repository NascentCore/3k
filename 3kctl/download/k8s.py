#!/usr/bin/env python3
import hashlib

from kubernetes import client
from kubernetes.client import ApiException


def resource_hash(hub, resource_id):
    return hashlib.sha1(("%s/%s" % (hub, resource_id)).encode("utf-8")).hexdigest()[:16]


def create_crd_name(hub, resource_id, resource_type):
    if resource_type == "model":
        return "model-storage-{0}".format(resource_hash(hub, resource_id))
    elif resource_type == "dataset":
        return "dataset-storage-{0}".format(resource_hash(hub, resource_id))


def create_pvc_name(hub, resource_id, resource_type):
    if resource_type == "model":
        return "pvc-model-{0}".format(resource_hash(hub, resource_id))
    elif resource_type == "dataset":
        return "pvc-dataset-{0}".format(resource_hash(hub, resource_id))


def create_job_name(hub, resource_id, resource_type):
    if resource_type == "model":
        return "download-model-{0}".format(resource_hash(hub, resource_id))
    elif resource_type == "dataset":
        return "download-dataset-{0}".format(resource_hash(hub, resource_id))


def create_crd_name_with_hash(resource_type, hashid):
    if resource_type == "model":
        return "model-storage-{0}".format(hashid)
    elif resource_type == "dataset":
        return "dataset-storage-{0}".format(hashid)


def create_pvc_name_with_hash(resource_type, hashid):
    if resource_type == "model":
        return "pvc-model-{0}".format(hashid)
    elif resource_type == "dataset":
        return "pvc-dataset-{0}".format(hashid)


def create_job_name_with_hash(resource_type, hashid):
    if resource_type == "model":
        return "download-model-{0}".format(hashid)
    elif resource_type == "dataset":
        return "download-dataset-{0}".format(hashid)


def get_crd_object(api_instance, group, version, resource, namespace, name):
    try:
        obj = api_instance.get_namespaced_custom_object(group, version, namespace, resource, name)
        return obj
    except Exception as e:
        raise e


def get_crd_objects(api_instance, group, version, resource, namespace):
    try:
        object_list = api_instance.list_namespaced_custom_object(group, version, namespace, resource)
        return object_list['items']
    except Exception as e:
        raise e


def create_pvc(api_instance, namespace, pvc_name, storage_size,
               access_mode="ReadWriteMany", pvc_type="dynamic", volume_name=""):
    body = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": namespace},
        "spec": {
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": storage_size}},
            "storageClassName": get_storage_class_from_configmap("cpod-system"),
            "volumeMode:": "Filesystem"
        },
    }
    if pvc_type == "static":
        body["spec"]["volumeName"] = volume_name

    try:
        api_response = api_instance.create_namespaced_persistent_volume_claim(
            namespace=namespace, body=body
        )
        print("PVC %s in namespace %s created" % (pvc_name, namespace))
        return api_response
    except ApiException as e:
        print("Error creating PVC: %s" % e)
        raise e


def create_download_job(api_instance, job_name, container_name, image, pvc_name, args, proxy_env="", namespace="public",
                        secret="aliyun-enterprise-registry", service_account_name="sa-downloader"):
    # 创建 Job 的配置
    job = client.V1Job(api_version="batch/v1", kind="Job",
                       metadata=client.V1ObjectMeta(name=job_name, namespace=namespace))
    container = client.V1Container(name=container_name, image=image, args=args)

    # 设置代理
    if proxy_env != "":
        container.env = [
            client.V1EnvVar(name="http_proxy", value=proxy_env),
            client.V1EnvVar(name="https_proxy", value=proxy_env),
            client.V1EnvVar(name="no_proxy", value="localhost,127.0.0.1,$(KUBERNETES_SERVICE_HOST)")
        ]

    # 定义卷挂载
    volume_mount = client.V1VolumeMount(name="data-volume", mount_path="/data")
    container.volume_mounts = [volume_mount]

    # 定义 Volume
    volume = client.V1Volume(name="data-volume",
                             persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name))
    pod_spec = client.V1PodSpec(restart_policy="Never", containers=[container], volumes=[volume])

    # 引用 Secret
    image_pull_secret = client.V1LocalObjectReference(name=secret)
    pod_spec.image_pull_secrets = [image_pull_secret]

    # 定义service account
    pod_spec.service_account_name = service_account_name

    # 定义 Job 规范
    job.spec = client.V1JobSpec(template=client.V1PodTemplateSpec(spec=pod_spec))
    job.spec.completions = 1
    job.spec.parallelism = 1

    try:
        # 创建 Job
        api_response = api_instance.create_namespaced_job(namespace=namespace, body=job)
        print(f"Job '{job_name}' created.")
    except ApiException as e:
        print(f"Exception when creating Job '{job_name}': {e}\n")
        raise e


def delete_job(api_instance, namespace, job_name):
    try:
        api_instance.delete_namespaced_job(name=job_name, namespace=namespace, body=client.V1DeleteOptions())
    except Exception as e:
        raise e


def delete_pvc(api_instance, namespace, pvc_name):
    try:
        api_instance.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace,
                                                               body=client.V1DeleteOptions())
    except Exception as e:
        raise e


def create_custom_resource(api_instance, group, version, kind, plural, name, namespace, spec):
    crd_instance = {
        "apiVersion": f"{group}/{version}",
        "kind": kind,
        "metadata": {
            "name": name,
            "namespace": namespace
        },
        "spec": spec
    }

    try:
        api_instance.create_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,  # 根据需要指定命名空间
            plural=plural,
            body=crd_instance
        )
    except ApiException as e:
        print(f"Exception when creating Custom Resource: {e}")
        raise e


def delete_custom_resource(api_instance, group, version, plural, name, namespace):
    try:
        api_instance.delete_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,  # Specify the namespace of the CR
            plural=plural,
            name=name
        )
    except ApiException as e:
        print(f"Exception when deleting Custom Resource: {e}")
        raise e


def delete_pods_for_job(api_instance, namespace, job_name):
    try:
        # List all pods in the specified namespace
        pods = api_instance.list_namespaced_pod(namespace, label_selector=f"job-name={job_name}")
        for pod in pods.items:
            api_instance.delete_namespaced_pod(pod.metadata.name, namespace)

    except ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod or delete_namespaced_pod: {e}")

def get_storage_class_from_configmap(namespace):
    api_instance = client.CoreV1Api()
    configmap = api_instance.read_namespaced_config_map(name="cpod-info", namespace=namespace)
    return configmap.data.get("storage_class", "ceph-filesystem")

def create_persistent_volume(api_instance, model_type, model_id, volume_name, hashid):
    # 定义PersistentVolume的元数据和规格
    pv = client.V1PersistentVolume(
        api_version="v1",
        kind="PersistentVolume",
        metadata=client.V1ObjectMeta(name=volume_name),
        spec=client.V1PersistentVolumeSpec(
            capacity={"storage": "100Mi"},
            access_modes=["ReadWriteMany"],
            storage_class_name=get_storage_class_from_configmap('cpod-system'),
            mount_options=[f'subdir=/{model_type}s/{model_id}'],
            csi=client.V1CSIPersistentVolumeSource(
                driver="csi.juicefs.com",
                volume_handle=f"{model_type}-{hashid}",
                node_publish_secret_ref=client.V1SecretReference(
                    name="juicefs-sc-secret",
                    namespace="kube-system"
                )
            )
        )
    )

    # 调用API来创建PV
    try:
        api_response = api_instance.create_persistent_volume(body=pv)
        print(f"Created PV: {api_response.metadata.name}")
    except client.ApiException as e:
        print("Error creating PV:", e)