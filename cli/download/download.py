#!/usr/bin/env python3

import hashlib
import math

from kubernetes import client, config
from kubernetes.client import ApiException
from kubernetes.client.models import V1ObjectMeta, V1PersistentVolumeClaimSpec, V1ResourceRequirements, \
    V1PersistentVolumeClaim
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

        pvc = pvc_name(model_id)
        config.load_kube_config()
        v1 = client.CoreV1Api()

        # ret = v1.list_persistent_volume_claim_for_all_namespaces(watch=False)
        # for i in ret.items:
        #     if i.metadata.name == pvc:
        #         print("PVC %s for model_id %s already exists" % (pvc, model_id))
        #         return

        # 创建PVC
        storage = model_size * 1.25
        if model_size > 1024:
            storage = "%dGi" % math.ceil(model_size / 1024)
        else:
            storage = "%dMi" % math.ceil(model_size)

        try:
            create_pvc(v1, "cpod", pvc, storage)
        except ApiException as e:
            print("create_pvc exception: %s" % e)
            return

        # 开启下载Job




def hub_factory(hub_name):
    if hub_name == "modelscope":
        return ModelScopeHub()
    else:
        return None


def pvc_name(model_id):
    hash_sha1 = hashlib.sha1(model_id.encode("utf-8"))
    return "model-{0}".format(hash_sha1.hexdigest()[:16])


# def create_ceph_pvc(api, namespace, name, storage):
#     body = V1PersistentVolumeClaim(api_version="v1", kind="PersistentVolumeClaim",
#                                    metadata=V1ObjectMeta(namespace=namespace, name=name),
#                                    spec=V1PersistentVolumeClaimSpec(
#                                        access_modes=["ReadWriteMany"],
#                                        resources=V1ResourceRequirements(requests={"storage": storage}),
#                                        storage_class_name="ceph-filesystem",
#                                        volume_mode="Filesystem"
#                                    ))
#     resp = api.create_namespaced_persistent_volume_claim(namespace, body=body)
#     if isinstance(resp, V1PersistentVolumeClaim):
#         print("create pvc %s ok" % name)
#     else:
#         print(resp)
#         raise Exception()


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
        print("PVC created. Status: %s" % str(api_response.status))
        return api_response
    except ApiException as e:
        print("Error creating PVC: %s" % e)
        raise e