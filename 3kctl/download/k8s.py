#!/usr/bin/env python3
import hashlib

from kubernetes import client
from kubernetes.client import ApiException


def model_hash(hub, model_id):
    return hashlib.sha1(("%s/%s" % (hub, model_id)).encode("utf-8")).hexdigest()[:16]


def create_crd_name(hub, model_id):
    return "model-storage-{0}".format(model_hash(hub, model_id))


def create_pvc_name(hub, model_id):
    return "pvc-model-{0}".format(model_hash(hub, model_id))


def create_job_name(hub, model_id):
    return "download-model-{0}".format(model_hash(hub, model_id))


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


def create_pvc(api_instance, namespace, pvc_name, storage_size):
    body = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": namespace},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": storage_size}},
            "storageClassName": "ceph-filesystem",
            "volumeMode:": "Filesystem"
        },
    }

    try:
        api_response = api_instance.create_namespaced_persistent_volume_claim(
            namespace=namespace, body=body
        )
        print("PVC %s in namespace %s created" % (pvc_name, namespace))
        return api_response
    except ApiException as e:
        print("Error creating PVC: %s" % e)
        raise e


def create_download_job(api_instance, job_name, container_name, image, pvc_name, args, namespace="cpod",
                        secret="aliyun-enterprise-registry", service_account_name="sa-downloader"):
    # 创建 Job 的配置
    job = client.V1Job(api_version="batch/v1", kind="Job",
                       metadata=client.V1ObjectMeta(name=job_name, namespace=namespace))
    container = client.V1Container(name=container_name, image=image, args=args)

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
        print(f"Custom Resource '{name}' created.")
    except ApiException as e:
        print(f"Exception when creating Custom Resource: {e}")
        raise e