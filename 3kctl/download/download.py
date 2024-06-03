#!/usr/bin/env python3

import math, os

from colorama import Fore, Style
from kubernetes import config
from plumbum import cli

from .k8s import *
from .resource import *

DOWNLOADER_VERSION = "v0.0.7"

GROUP = "cpod.cpod"
VERSION = "v1"
MODEL_PLURAL = "modelstorages"
DATASET_PLURAL = "datasetstorages"

OSS_ENDPOINT = os.getenv('OSS_ENDPOINT', 'https://oss-cn-beijing.aliyuncs.com')
OSS_ACCESS_ID = os.getenv('OSS_ACCESS_ID')
OSS_ACCESS_KEY = os.getenv('OSS_ACCESS_KEY')
OSS_BUCKET = 'sxwl-cache'

class Download(cli.Application):
    """Download model or dataset"""


@Download.subcommand("model")
class Model(cli.Application):
    """download model"""

    def main(self, hub_name, model_id, proxy="", depth=1, downloader_version=DOWNLOADER_VERSION, namespace="public"):
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub:{0} is not supported".format(hub_name))
            return

        # check the model id exists
        if not hub.have_model(model_id):
            print("hub:%s model:%s dose not exist" % (hub_name, model_id))
            return

        depth = int(depth)
        model_size = math.ceil(hub.model_size(model_id))  # get the model size in GB
        crd_name = create_crd_name(hub_name, model_id, "model")
        pvc_name = create_pvc_name(hub_name, model_id, "model")
        job_name = create_job_name(hub_name, model_id, "model")
        if hub_name == "oss":
            crd_name = model_crd_name(resource_to_oss_path(MODEL, model_id))
            pvc_name = model_pvc_name(resource_to_oss_path(MODEL, model_id))
            job_name = model_download_job_name(resource_to_oss_path(MODEL, model_id))

        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
        batch_v1_api = client.BatchV1Api()
        custom_objects_api = client.CustomObjectsApi()

        # check if crd already exists
        try:
            crd_obj = get_crd_object(custom_objects_api, GROUP, VERSION, MODEL_PLURAL, namespace, crd_name)
            if 'status' in crd_obj:
                phase_value = crd_obj['status'].get('phase', None)
                if phase_value is not None:
                    if phase_value == "downloading":
                        print("hub:%s model_id:%s crd:%s is downloading" % (hub_name, model_id, crd_name))
                    elif phase_value == "done":
                        print("hub:%s model_id:%s already crd:%s downloaded" % (hub_name, model_id, crd_name))
                    else:
                        print("hub:%s model_id:%s phase:%s crd:%s please check the phase" %
                              (hub_name, model_id, phase_value, crd_name))
                else:
                    print('''hub:%s model_id:%s please try again later, if this continue occurs, please delete the crd:
    kubectl delete ModelStorage %s -n %s''' % (hub_name, model_id, crd_name, namespace))
            else:
                print('''crd %s exists and without status. You can delete it by:
    kubectl delete ModelStorage %s -n %s''' % (crd_name, crd_name, namespace))
            return
        except ApiException as e:
            if e.status == 404:
                # crd 不存在，那么如果有job或pvc，都应该清理掉
                try:
                    delete_job(batch_v1_api, namespace, job_name)
                except ApiException as e:
                    pass
                try:
                    delete_pvc(core_v1_api, namespace, pvc_name)
                except ApiException as e:
                    pass
            else:
                print("get_crd_object exception %s" % e)
                return

        storage = "100Mi"
        job_pvc_name = 'model-download'
        try:
            create_persistent_volume(core_v1_api, 'model', '', job_pvc_name, job_pvc_name)
            create_pvc(core_v1_api, namespace, job_pvc_name, storage, pvc_type="static", volume_name=job_pvc_name)
            create_persistent_volume(core_v1_api, 'model', model_id, pvc_name,
                                     hash_data(resource_to_oss_path(MODEL, model_id)))
            create_pvc(core_v1_api, namespace, pvc_name, storage, pvc_type="static", volume_name=pvc_name)
        except ApiException as e:
            print("create_persistent_volume exception: %s" % e)
            return

        # 创建下载Job
        try:
            job_params = [
                "-o", f"/data/{model_id}",
                "-g", GROUP,
                "-v", VERSION,
                "-p", MODEL_PLURAL,
                "-n", namespace,
                "--name", crd_name
            ]

            if hub_name == "oss":
                additional_params = [
                    "oss", hub.model_url(model_id),
                    "-t", str(model_size * 1024 * 1024 * 1024),
                    "--endpoint", OSS_ENDPOINT,
                    "--access_id", OSS_ACCESS_ID,
                    "--access_key", OSS_ACCESS_KEY
                ]
                job_params.extend(additional_params)
            else:
                additional_params = [
                    "git", hub.model_url(model_id),
                    "-d", str(depth),
                    "-t", str(model_size * 2 * 1024 * 1024 * 1024)
                ]
                job_params.extend(additional_params)

            create_download_job(batch_v1_api,
                                job_name,
                                "model-downloader",
                                "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:%s" % downloader_version,
                                job_pvc_name,
                                job_params,
                                proxy,
                                namespace,
                                "aliyun-enterprise-registry",
                                "sa-downloader")
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
                plural=MODEL_PLURAL,
                name=crd_name,
                spec={
                    "modeltype": hub_name,
                    "modelname": model_id,
                    "pvc": pvc_name,
                },
                namespace=namespace
            )
            print(f"Custom Resource '{crd_name}' created.")
        except ApiException as e:
            print("create_crd_record exception: %s" % e)
            return


@Download.subcommand("dataset")
class Dataset(cli.Application):
    """download dataset"""

    def main(self, hub_name, dataset_id, proxy="", depth=1, downloader_version=DOWNLOADER_VERSION, namespace="public"):
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub:{0} is not supported".format(hub_name))
            return

        # check the dataset id exists
        if not hub.have_dataset(dataset_id):
            print("hub:%s dataset:%s dose not exist" % (hub_name, dataset_id))
            return

        depth = int(depth)
        dataset_size = math.ceil(hub.dataset_size(dataset_id))  # get the dataset size in GB
        crd_name = create_crd_name(hub_name, dataset_id, "dataset")
        pvc_name = create_pvc_name(hub_name, dataset_id, "dataset")
        job_name = create_job_name(hub_name, dataset_id, "dataset")
        if hub_name == "oss":
            crd_name = dataset_crd_name(resource_to_oss_path(DATASET, dataset_id))
            pvc_name = dataset_pvc_name(resource_to_oss_path(DATASET, dataset_id))
            job_name = dataset_download_job_name(resource_to_oss_path(DATASET, dataset_id))
        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
        batch_v1_api = client.BatchV1Api()
        custom_objects_api = client.CustomObjectsApi()

        # check if crd already exists
        try:
            crd_obj = get_crd_object(custom_objects_api, GROUP, VERSION, DATASET_PLURAL, namespace, crd_name)
            if 'status' in crd_obj:
                phase_value = crd_obj['status'].get('phase', None)
                if phase_value is not None:
                    if phase_value == "downloading":
                        print("hub:%s dataset_id:%s is downloading" % (hub_name, dataset_id))
                    elif phase_value == "done":
                        print("hub:%s dataset_id:%s already downloaded" % (hub_name, dataset_id))
                    else:
                        print("hub:%s dataset_id:%s phase:%s please check the phase" % (
                            hub_name, dataset_id, phase_value))
                else:
                    print('''hub:%s dataset_id:%s please try again later, if this continue occurs, please delete the crd:
    kubectl delete DataSetStorage %s -n %s''' % (hub_name, dataset_id, crd_name, namespace))
            else:
                print('''crd %s exists and without status. You can delete it by:
    kubectl delete DataSetStorage %s -n %s''' % (crd_name, crd_name, namespace))
            return
        except ApiException as e:
            if e.status == 404:
                # crd 不存在，那么如果有job或pvc，都应该清理掉
                try:
                    delete_job(batch_v1_api, namespace, job_name)
                except ApiException as e:
                    pass
                try:
                    delete_pvc(core_v1_api, namespace, pvc_name)
                except ApiException as e:
                    pass
            else:
                print("get_crd_object exception %s" % e)
                return

        storage = "100Mi"
        job_pvc_name = 'dataset-download'
        try:
            create_persistent_volume(core_v1_api, 'dataset', '', job_pvc_name, job_pvc_name)
            create_pvc(core_v1_api, namespace, job_pvc_name, storage, pvc_type="static", volume_name=job_pvc_name)
            create_persistent_volume(core_v1_api, 'dataset', dataset_id, pvc_name,
                                     hash_data(resource_to_oss_path(MODEL, dataset_id)))
            create_pvc(core_v1_api, namespace, pvc_name, storage, pvc_type="static", volume_name=pvc_name)
        except ApiException as e:
            print("create_persistent_volume exception: %s" % e)
            return

        # 创建下载Job
        try:
            job_params = [
                "-o", f"/data/{dataset_id}",
                "-g", GROUP,
                "-v", VERSION,
                "-p", DATASET_PLURAL,
                "-n", namespace,
                "--name", crd_name
            ]

            if hub_name == "oss":
                additional_params = [
                    "oss", hub.dataset_url(dataset_id),
                    "-t", str(dataset_size * 1024 * 1024 * 1024),
                    "--endpoint", OSS_ENDPOINT,
                    "--access_id", OSS_ACCESS_ID,
                    "--access_key", OSS_ACCESS_KEY
                ]
                job_params.extend(additional_params)
            else:
                additional_params = [
                    "git", hub.dataset_url(dataset_id),
                    "-d", str(depth),
                    "-t", str(dataset_id * 2 * 1024 * 1024 * 1024)
                ]
                job_params.extend(additional_params)

            create_download_job(batch_v1_api,
                                job_name,
                                "dataset-downloader",
                                "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:%s" % downloader_version,
                                job_pvc_name,
                                job_params,
                                proxy,
                                namespace,
                                "aliyun-enterprise-registry",
                                "sa-downloader")
        except ApiException as e:
            print("create_download_job exception: %s" % e)
            return

        # 创建CRD对象
        try:
            create_custom_resource(
                api_instance=custom_objects_api,
                group=GROUP,
                version=VERSION,
                kind="DataSetStorage",
                plural=DATASET_PLURAL,
                name=crd_name,
                spec={
                    "datasettype": hub_name,
                    "datasetname": dataset_id,
                    "pvc": pvc_name,
                },
                namespace=namespace
            )
            print(f"Custom Resource '{crd_name}' created.")
        except ApiException as e:
            print("create_crd_record exception: %s" % e)
            return


@Download.subcommand("adapter")
class Adapter(cli.Application):
    """download adapter"""

    def main(self, hub_name, adapter_id, proxy="", depth=1, downloader_version=DOWNLOADER_VERSION, namespace="public"):
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub:{0} is not supported".format(hub_name))
            return

        # check the adapter id exists
        if not hub.have_adapter(adapter_id):
            print("hub:%s adapter:%s dose not exist" % (hub_name, adapter_id))
            return

        depth = int(depth)
        adapter_size = math.ceil(hub.adapter_size(adapter_id))  # get the dataset size in GB
        crd_name = create_crd_name(hub_name, adapter_id, "adapter")
        pvc_name = create_pvc_name(hub_name, adapter_id, "adapter")
        job_name = create_job_name(hub_name, adapter_id, "adapter")
        if hub_name == "oss":
            crd_name = model_crd_name(resource_to_oss_path(ADAPTER, adapter_id))
            pvc_name = model_pvc_name(resource_to_oss_path(ADAPTER, adapter_id))
            job_name = model_download_job_name(resource_to_oss_path(ADAPTER, adapter_id))
        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
        batch_v1_api = client.BatchV1Api()
        custom_objects_api = client.CustomObjectsApi()

        # check if crd already exists
        try:
            crd_obj = get_crd_object(custom_objects_api, GROUP, VERSION, MODEL_PLURAL, namespace, crd_name)
            if 'status' in crd_obj:
                phase_value = crd_obj['status'].get('phase', None)
                if phase_value is not None:
                    if phase_value == "downloading":
                        print("hub:%s adapter_id:%s is downloading" % (hub_name, adapter_id))
                    elif phase_value == "done":
                        print("hub:%s adapter_id:%s already downloaded" % (hub_name, adapter_id))
                    else:
                        print("hub:%s adapter_id:%s phase:%s please check the phase" % (
                            hub_name, adapter_id, phase_value))
                else:
                    print('''hub:%s adapter_id:%s please try again later, if this continue occurs, please delete the crd:
    kubectl delete ModelStorage %s -n %s''' % (hub_name, adapter_id, crd_name, namespace))
            else:
                print('''crd %s exists and without status. You can delete it by:
    kubectl delete ModelStorage %s -n %s''' % (crd_name, crd_name, namespace))
            return
        except ApiException as e:
            if e.status == 404:
                # crd 不存在，那么如果有job或pvc，都应该清理掉
                try:
                    delete_job(batch_v1_api, namespace, job_name)
                except ApiException as e:
                    pass
                try:
                    delete_pvc(core_v1_api, namespace, pvc_name)
                except ApiException as e:
                    pass
            else:
                print("get_crd_object exception %s" % e)
                return

        storage = "100Mi"
        job_pvc_name = 'adapter-download'
        try:
            create_persistent_volume(core_v1_api, 'adapter', '', job_pvc_name, job_pvc_name)
            create_pvc(core_v1_api, namespace, job_pvc_name, storage, pvc_type="static", volume_name=job_pvc_name)
            create_persistent_volume(core_v1_api, 'adapter', adapter_id, pvc_name,
                                     hash_data(resource_to_oss_path(MODEL, adapter_id)))
            create_pvc(core_v1_api, namespace, pvc_name, storage, pvc_type="static", volume_name=pvc_name)
        except ApiException as e:
            print("create_persistent_volume exception: %s" % e)
            return

        # 创建下载Job
        try:
            job_params = [
                "-o", f"/data/{adapter_id}",
                "-g", GROUP,
                "-v", VERSION,
                "-p", MODEL_PLURAL,
                "-n", namespace,
                "--name", crd_name
            ]

            if hub_name == "oss":
                additional_params = [
                    "oss", hub.adapter_url(adapter_id),
                    "-t", str(adapter_size * 1024 * 1024 * 1024),
                    "--endpoint", OSS_ENDPOINT,
                    "--access_id", OSS_ACCESS_ID,
                    "--access_key", OSS_ACCESS_KEY
                ]
                job_params.extend(additional_params)
            else:
                additional_params = [
                    "git", hub.dataset_url(adapter_id),
                    "-d", str(depth),
                    "-t", str(adapter_id * 2 * 1024 * 1024 * 1024)
                ]
                job_params.extend(additional_params)

            create_download_job(batch_v1_api,
                                job_name,
                                "dataset-downloader",
                                "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:%s" % downloader_version,
                                job_pvc_name,
                                job_params,
                                proxy,
                                namespace,
                                "aliyun-enterprise-registry",
                                "sa-downloader")
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
                plural=MODEL_PLURAL,
                name=crd_name,
                spec={
                    "datasettype": hub_name,
                    "datasetname": adapter_id,
                    "pvc": pvc_name,
                },
                namespace=namespace
            )
            print(f"Custom Resource '{crd_name}' created.")
        except ApiException as e:
            print("create_crd_record exception: %s" % e)
            return


@Download.subcommand("status")
class Status(cli.Application):
    """download status check all download jobs status"""

    def main(self, namespace="public"):
        config.load_kube_config()
        custom_objects_api = client.CustomObjectsApi()
        try:
            model_storage_objs = get_crd_objects(custom_objects_api, GROUP, VERSION, MODEL_PLURAL, namespace)
            dataset_storage_objs = get_crd_objects(custom_objects_api, GROUP, VERSION, DATASET_PLURAL, namespace)
        except ApiException as e:
            print("get_crd_objects exception: %s", e)
            return

        data_list = [["HUB", "MODEL_ID", "HASH", "PHASE"]]
        for model_storage_obj in model_storage_objs:
            spec = model_storage_obj.get('spec', {})
            status = model_storage_obj.get('status', {})
            model_hash = resource_hash(spec.get('modeltype'), spec.get('modelname'))
            if spec.get('modeltype') == "oss":
                model_hash = hash_data(resource_to_oss_path(MODEL, spec.get('modelname')))
            data_list.append([spec.get('modeltype'),
                              spec.get('modelname'),
                              model_hash,
                              status.get("phase", "nil")])
        print('\n')
        print_list(data_list)

        data_list = [["HUB", "DATASET_ID", "HASH", "PHASE"]]
        for dataset_storage_obj in dataset_storage_objs:
            spec = dataset_storage_obj.get('spec', {})
            status = dataset_storage_obj.get('status', {})
            dataset_hash = resource_hash(spec.get('datasettype'), spec.get('datasetname'))
            if spec.get('datasettype') == "oss":
                dataset_hash = hash_data(resource_to_oss_path(DATASET, spec.get('datasetname')))
            data_list.append([spec.get('datasettype'),
                              spec.get('datasetname'),
                              dataset_hash,
                              status.get("phase", "nil")])
        print('\n')
        print_list(data_list)


@Download.subcommand("delete")
class Delete(cli.Application):
    """delete model or dataset and all related resource"""

    def main(self, resource, hashid, namespace="public"):
        if resource not in ["model", "dataset"]:
            print("Error: Resource must be 'model' or 'dataset'.")
            return 1  # Exit with an error code

        if cli.terminal.ask(f"确认删除在namespace: {namespace} 中的 {resource}: {hashid} ?", default=False):
            print(f"Deleting resources in namespace {namespace}...")

            config.load_kube_config()
            core_v1_api = client.CoreV1Api()
            batch_v1_api = client.BatchV1Api()
            custom_objects_api = client.CustomObjectsApi()

            crd_name = create_crd_name_with_hash(resource, hashid)
            pvc_name = create_pvc_name_with_hash(resource, hashid)
            job_name = create_job_name_with_hash(resource, hashid)

            try:
                # Delete the job
                delete_job(batch_v1_api, namespace, job_name)
                print(f"Job '{job_name}' deleted successfully.")

                # Delete the pods of job
                delete_pods_for_job(core_v1_api, namespace, job_name)
                print(f"Pods of Job '{job_name}' deleted successfully.")

                # Delete the PVC
                delete_pvc(core_v1_api, namespace, pvc_name)
                print(f"PVC '{pvc_name}' deleted successfully.")

                # Determine the plural form of the resource and delete the custom resource
                plural = MODEL_PLURAL if resource == "model" else DATASET_PLURAL
                delete_custom_resource(custom_objects_api, GROUP, VERSION, plural, crd_name, namespace)
                print(f"CR '{crd_name}': '{crd_name}' deleted successfully.")

            except ApiException as e:
                print(f"Delete exception: {e}")
        else:
            print("Deletion canceled.")


def hub_factory(hub_name):
    if hub_name == "modelscope":
        from .model_scope import ModelScopeHub
        return ModelScopeHub()
    if hub_name == "huggingface":
        from .hugging_face import HuggingFaceHub
        return HuggingFaceHub()
    if hub_name == "oss":
        from .oss import OSSHub
        return OSSHub(OSS_ENDPOINT, OSS_ACCESS_ID, OSS_ACCESS_KEY, OSS_BUCKET)
    else:
        return None


def print_list(data_list):
    # 计算每一列的最大宽度
    column_widths = [max(len(str(item)) for item in column) for column in zip(*data_list)]

    # 打印表头
    header = "|".join(" {:<{}} ".format(header, width) for header, width in zip(data_list[0], column_widths))
    print(header)
    print("-" * len(header))

    # 打印数据行
    for row in data_list[1:]:
        # 根据第四列的值设置颜色
        phase_value = row[3]
        if phase_value == "fail":
            colored_phase = f"{Fore.RED}{phase_value}{Style.RESET_ALL}"
        elif phase_value == "done":
            colored_phase = f"{Fore.GREEN}{phase_value}{Style.RESET_ALL}"
        elif phase_value == "downloading":
            colored_phase = f"{Fore.YELLOW}{phase_value}{Style.RESET_ALL}"
        else:
            colored_phase = phase_value

        # 格式化并打印数据行
        row_str = "|".join(" {:<{}} ".format(cell, width) for cell, width in
                           zip([row[0], row[1], row[2], colored_phase], column_widths))
        print(row_str)
