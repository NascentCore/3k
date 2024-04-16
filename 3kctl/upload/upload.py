#!/usr/bin/env python3
from plumbum import cli
from .utils import *


class Upload(cli.Application):
    """upload model or dataset to registry"""


class UploadBase(cli.Application):
    """Base class for uploading to registry"""
    name = cli.SwitchAttr("--name", str, mandatory=True, help="resource name")
    dir = cli.SwitchAttr("--dir", str, mandatory=True, help="resource directory")
    delete_existing = cli.Flag("--delete-existing", default=False, help="delete existing resource storage and pvc if exists")

    def execute(self, data_type):
        print(self.name, self.dir, self.delete_existing, data_type)
        models = {
            "ZhipuAI/chatglm3-6b": "alpaca",
            "meta-llama/Llama-2-7b": "alpaca",
            "baichuan-inc/Baichuan2-7B-Chat": "alpaca",
            "IDEA-CCNL/Ziya-LLaMA-13B-v1": "alpaca",
            "google/gemma-2b-it": "gemma",
            "mistralai/Mistral-7B-v0.1": "mistral",
            "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
            "mistralai/Mixtral-8x7B-v0.1": "mistral",
            "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral"
        } 
            
        namespace = 'cpod'
        model_type = "public" if self.name in models else "user-"
        hashed_name = get_hashed_name(f"{data_type}s/{model_type}/{self.name}")
        pvc_name = f'pvc-{data_type}-{hashed_name}'
        pod_name = f'{data_type}-copy-pod'
        crd_name = f'{data_type}-storage-{hashed_name}'
        api_version = 'cpod.cpod/v1'
        crd_plural = f"{data_type}storages"

        # 检查PVC是否存在
        if check_pvc_exists(namespace, pvc_name):
            print(f"PVC {pvc_name} already exists in namespace {namespace}. Exiting.")
            if self.delete_existing:
                delete_crd_and_pvc(namespace, "pvc", pvc_name, api_version, crd_plural)
            else:
                return

        # 检查CRD是否存在
        if check_crd_exists(namespace, crd_name, api_version, crd_plural):
            print(f"CRD {crd_name} already exists in namespace {namespace}. Exiting.")
            if self.delete_existing:
                delete_crd_and_pvc(namespace, "crd", crd_name, api_version, crd_plural)
            else:
                return

        # 如果PVC和CRD都不存在，则继续执行创建过程
        dir_size = get_dir_size(self.dir)
        print(f"Directory size: {dir_size} bytes")

        dir_size = get_dir_size(self.dir)
        print(f"Directory size: {dir_size} bytes")

        create_pvc(namespace, pvc_name, dir_size)
        create_pod_with_pvc(namespace, pod_name, pvc_name)

        wait_for_pod_ready(namespace, pod_name)
        copy_to_pvc(namespace, pod_name, self.dir)
        create_crd(namespace, data_type, self.name, crd_name, pvc_name, api_version)
        delete_pod(namespace, pod_name)
        update_crd_status(namespace, crd_name, data_type, api_version, dir_size, 'done', models.get(self.name, ""))


@Upload.subcommand("model")
class Model(UploadBase):
    """upload model to registry"""

    def main(self):
        self.execute("model")

@Upload.subcommand("dataset")
class Dataset(UploadBase):
    """upload dataset to registry"""

    def main(self):
        self.execute("dataset")