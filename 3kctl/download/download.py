#!/usr/bin/env python3

import math

from colorama import Fore, Style
from kubernetes import config
from plumbum import cli

from .hugging_face import HuggingFaceHub
from .k8s import *
from .model_scope import ModelScopeHub

GROUP = "cpod.sxwl.ai"
VERSION = "v1"
PLURAL = "modelstorages"


class Download(cli.Application):
    """Download model or dataset"""


@Download.subcommand("model")
class Model(cli.Application):
    """download model"""

    def main(self, hub_name, model_id, downloader_version="v0.0.1", namespace="cpod"):
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub:{0} is not supported".format(hub_name))
            return

        # check the model id exists
        if not hub.have_model(model_id):
            print("hub:%s model:%s dose not exist" % (hub_name, model_id))
            return

        model_size = math.ceil(hub.size(model_id))  # get the model size in GB
        crd_name = create_crd_name(hub_name, model_id)
        pvc_name = create_pvc_name(hub_name, model_id)
        job_name = create_job_name(hub_name, model_id)
        config.load_kube_config()
        core_v1_api = client.CoreV1Api()
        batch_v1_api = client.BatchV1Api()
        custom_objects_api = client.CustomObjectsApi()

        # check if crd already exists
        try:
            crd_obj = get_crd_object(custom_objects_api, GROUP, VERSION, PLURAL, namespace, crd_name)
            if 'status' in crd_obj:
                phase_value = crd_obj['status'].get('phase', None)
                if phase_value is not None:
                    if phase_value == "downloading":
                        print("hub:%s model_id:%s is downloading" % (hub_name, model_id))
                    elif phase_value == "done":
                        print("hub:%s model_id:%s already downloaded" % (hub_name, model_id))
                    else:
                        print("hub:%s model_id:%s phase:%s please check the phase" % (hub_name, model_id, phase_value))
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

        # 创建PVC
        storage = model_size * 3  # 申请3倍model_size的空间
        storage = "%dGi" % math.ceil(storage)
        print("model {0} size {1} GB pvc {2} GB".format(hub_name, model_size, storage))

        try:
            create_pvc(core_v1_api, namespace, pvc_name, storage)
        except ApiException as e:
            print("create_pvc exception: %s" % e)
            return

        # 创建下载Job
        try:
            create_download_job(batch_v1_api,
                                job_name,
                                "model-downloader",
                                "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:%s" % downloader_version,
                                pvc_name,
                                ["git", hub.git_url(model_id),
                                 "-g", GROUP,
                                 "-v", VERSION,
                                 "-p", PLURAL,
                                 "-n", namespace,
                                 "--name", crd_name],
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
                plural=PLURAL,
                name=crd_name,
                spec={
                    "modeltype": hub_name,
                    "modelname": model_id,
                    "pvc": pvc_name,
                },
                namespace=namespace
            )
        except ApiException as e:
            print("create_crd_record exception: %s" % e)
            return


@Download.subcommand("status")
class Status(cli.Application):
    """download status check all download jobs status"""

    def main(self, namespace="cpod"):
        config.load_kube_config()
        custom_objects_api = client.CustomObjectsApi()
        try:
            model_storage_objs = get_crd_objects(custom_objects_api, GROUP, VERSION, PLURAL, namespace)
        except ApiException as e:
            print("get_crd_objects exception: %s", e)
            return

        data_list = [["HUB", "MODEL_ID", "HASH", "PHASE"]]
        for model_storage_obj in model_storage_objs:
            spec = model_storage_obj.get('spec', {})
            status = model_storage_obj.get('status', {})
            data_list.append([spec.get('modeltype'),
                              spec.get('modelname'),
                              model_hash(spec.get('modeltype'), spec.get('modelname')),
                              status.get("phase", "nil")])
        print('\n')
        print_list(data_list)


def hub_factory(hub_name):
    if hub_name == "modelscope":
        return ModelScopeHub()
    if hub_name == "huggingface":
        return HuggingFaceHub()
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