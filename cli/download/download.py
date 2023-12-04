#!/usr/bin/env python3

import hashlib
import math

from kubernetes import client, config
from kubernetes.client import ApiException
from plumbum import cli

from .model_scope import ModelScopeHub


class Download(cli.Application):
    """Download model or dataset"""


@Download.subcommand("model")
class Model(cli.Application):
    """download model"""

    def main(self, hub_name, model_id):
        # get the model size
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub {0} is not supported".format(hub_name))
            return

        model_size = math.ceil(hub.size(model_id))
        print("model {0} size {1} GB".format(hub_name, model_size))

        # todo 前置检查
        pvc = create_pvc_name(model_id)
        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
        batch_v1_api = client.BatchV1Api()

        # 创建PVC
        storage = model_size * 1.25
        if model_size > 1024:
            storage = "%dGi" % math.ceil(model_size / 1024)
        else:
            storage = "%dMi" % math.ceil(model_size)

        try:
            create_pvc(core_v1_api, "cpod", pvc, storage)
        except ApiException as e:
            print("create_pvc exception: %s" % e)
            return

        # 创建下载Job
        try:
            create_download_job(batch_v1_api,
                                "model-download",
                                "model-downloader",
                                "registry.cn-hangzhou.aliyuncs.com/sxwl-ai/downloader:v1.0.0",
                                pvc,
                                ["-s", hub.git_url(model_id)])
        except ApiException as e:
            print("create_download_job exception: %s" % e)
            return

        # 写CRD


def hub_factory(hub_name):
    if hub_name == "modelscope":
        return ModelScopeHub()
    else:
        return None


def create_pvc_name(model_id):
    hash_sha1 = hashlib.sha1(model_id.encode("utf-8"))
    return "model-{0}".format(hash_sha1.hexdigest()[:16])


def create_pvc(api_instance, namespace, pvc_name, storage_size):
    body = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": namespace},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": storage_size}},
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


def create_download_job(api_instance, job_name, container_name, image, pvc_name, args):
    # 创建 Job 的配置
    job = client.V1Job(api_version="batch/v1", kind="Job", metadata=client.V1ObjectMeta(name=job_name))
    container = client.V1Container(name=container_name, image=image, args=args)

    # 定义卷挂载
    volume_mount = client.V1VolumeMount(name="data-volume", mount_path="/data")
    container.volume_mounts = [volume_mount]

    # 定义 Volume
    volume = client.V1Volume(name="data-volume",
                             persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name))
    pod_spec = client.V1PodSpec(restart_policy="Never", containers=[container], volumes=[volume])

    # 定义 Job 规范
    job.spec = client.V1JobSpec(template=client.V1PodTemplateSpec(spec=pod_spec))
    job.spec.completions = 1
    job.spec.parallelism = 1

    try:
        # 创建 Job
        api_response = api_instance.create_namespaced_job(namespace="default", body=job)
        print(f"Job '{job_name}' created. status='{str(api_response.status)}'")
    except ApiException as e:
        print(f"Exception when creating Job '{job_name}': {e}\n")
        raise e
