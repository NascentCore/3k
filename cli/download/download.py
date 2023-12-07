#!/usr/bin/env python3

import hashlib
import math

from kubernetes import client, config
from kubernetes.client import ApiException
from plumbum import cli

from .model_scope import ModelScopeHub

GROUP = "cpod.sxwl.ai"
VERSION = "v1"
PLURAL = "modelstorages",


class Download(cli.Application):
    """Download model or dataset"""


@Download.subcommand("model")
class Model(cli.Application):
    """download model"""

    def main(self, hub_name, model_id, namespace="cpod"):
        # get the model size
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub {0} is not supported".format(hub_name))
            return

        model_size = math.ceil(hub.size(model_id))  # in GB
        print("model {0} size {1} GB".format(hub_name, model_size))

        crd_name = create_crd_name(hub_name, model_id)
        pvc = create_pvc_name(hub_name, model_id)
        job_name = create_job_name(hub_name, model_id)
        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
        batch_v1_api = client.BatchV1Api()
        custom_objects_api = client.CustomObjectsApi()

        # 创建PVC
        storage = model_size * 1.25
        storage = "%dGi" % math.ceil(storage)
        try:
            create_pvc(core_v1_api, namespace, pvc, storage)
        except ApiException as e:
            print("create_pvc exception: %s" % e)
            return

        # 创建下载Job
        try:
            create_download_job(batch_v1_api,
                                job_name,
                                "model-downloader",
                                "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:v2.0.0",
                                pvc,
                                ["git",
                                 "-s", hub.git_url(model_id),
                                 "-g", GROUP,
                                 "-v", VERSION,
                                 "-p", PLURAL,
                                 "-n", job_name,
                                 "--namespace", namespace],
                                namespace,
                                "aliyun-enterprise-registry")
        except ApiException as e:
            print("create_download_job exception: %s" % e)
            return

        # 创建CRD对象
        try:
            create_custom_resource(
                api_instance=custom_objects_api,
                group=GROUP,
                version=VERSION,
                kind="ModelStorage",
                plural=PLURAL,
                name=crd_name,
                spec={
                    "modeltype": hub_name,
                    "modelname": model_id,
                    "pvc": pvc,
                },
                namespace=namespace
            )
        except ApiException as e:
            print("create_crd_record exception: %s" % e)
            return

        # 更新CRD状态为downloading
        try:
            update_custom_resource_status(
                api_instance=custom_objects_api,
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                name=crd_name,
                namespace=namespace,
                new_phase="downloading"
            )
        except ApiException as e:
            print("create_crd_record exception: %s" % e)
            return


def hub_factory(hub_name):
    if hub_name == "modelscope":
        return ModelScopeHub()
    else:
        return None


def create_crd_name(hub, model_id):
    hash_sha1 = hashlib.sha1(("%s/%s" % (hub, model_id)).encode("utf-8"))
    return "model-storage-{0}".format(hash_sha1.hexdigest()[:16])


def create_pvc_name(hub, model_id):
    hash_sha1 = hashlib.sha1(("%s/%s" % (hub, model_id)).encode("utf-8"))
    return "pvc-model-{0}".format(hash_sha1.hexdigest()[:16])


def create_job_name(hub, model_id):
    hash_sha1 = hashlib.sha1(("%s/%s" % (hub, model_id)).encode("utf-8"))
    return "download-model-{0}".format(hash_sha1.hexdigest()[:16])


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
        print("PVC %s in namespace %s created. Status: %s" % (pvc_name, namespace, str(api_response.status)))
        return api_response
    except ApiException as e:
        print("Error creating PVC: %s" % e)
        raise e


def create_download_job(api_instance, job_name, container_name, image, pvc_name, args, namespace="default",
                        secret="aliyun-enterprise-registry"):
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

    # 定义 Job 规范
    job.spec = client.V1JobSpec(template=client.V1PodTemplateSpec(spec=pod_spec))
    job.spec.completions = 1
    job.spec.parallelism = 1

    try:
        # 创建 Job
        api_response = api_instance.create_namespaced_job(namespace=namespace, body=job)
        print(f"Job '{job_name}' created. status='{str(api_response.status)}'")
    except ApiException as e:
        print(f"Exception when creating Job '{job_name}': {e}\n")
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


def update_custom_resource_status(api_instance, group, version, plural, name, namespace, new_phase):
    try:
        # 获取当前对象
        current_object = api_instance.get_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name
        )

        # 检查是否存在 status 字段，如果不存在，则创建
        if "status" not in current_object:
            current_object["status"] = {}

        # 设置 status.phase 字段
        current_object["status"]["phase"] = new_phase

        # 替换对象的状态
        api_response = api_instance.patch_namespaced_custom_object_status(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name,
            body=current_object
        )
        print(f"Custom Resource '{name}' status updated. Status='{str(api_response)}'")
    except ApiException as e:
        print(f"Exception when updating Custom Resource status: {e}")
        raise e
