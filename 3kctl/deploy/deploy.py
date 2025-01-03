#!/usr/bin/env python3
from plumbum import cli
from .registry import *
from .kubernetes import *


class Deploy(cli.Application):
    """cluster offline deployment"""


@Deploy.subcommand("init")
class Init(cli.Application):
    """initialize module"""


@Init.subcommand("registry")
class Registry(cli.Application):
    """initialize the private repository"""

    def main(self):
        init_registry()


@Init.subcommand("cpodinfo")
class CpodInfo(cli.Application):
    """initialize cpod info"""
    access_key = cli.SwitchAttr("--access-key", str, mandatory=True, help="cloud.nascentcore.ai user accessKey")
    api_address = cli.SwitchAttr("--api-address", str, mandatory=False, default="https://cloud.nascentcore.ai", help="api address")
    storage_class = cli.SwitchAttr("--storage-class", str, mandatory=False, default="ceph-filesystem", help="storageClass name")
    oss_bucket = cli.SwitchAttr("--oss-bucket", str, mandatory=False, default="sxwl-ai", help="upload model to oss bucket name")
    log_level = cli.SwitchAttr("--log-level", str, mandatory=False, default="INFO", help="log level")

    def main(self):
        init_cpod_info(self.access_key, self.api_address, self.storage_class, self.oss_bucket, self.log_level)


@Deploy.subcommand("push")
class Push(cli.Application):
    """push images or helm charts to registry"""


@Push.subcommand("images")
class Images(cli.Application):
    """push images to registry"""

    def main(self):
        create_project()
        push_images()


@Push.subcommand("charts")
class Charts(cli.Application):
    """push helm charts to registry"""

    def main(self):
        install_helm_push()
        push_charts()


@Deploy.subcommand("install")
class Install(cli.Application):
    """install kubernetes cluster or operators"""
    online = cli.Flag("--online", help="Specify if installation should be done online")

    def _pass_online_flag(self, subcommand_instance):
        subcommand_instance.online = self.online


@Install.subcommand("kubernetes")
class Kubernetes(cli.Application):
    """install kubernetes cluster"""
    online = False

    def main(self):
        self.parent._pass_online_flag(self)
        if self.online:
            print("Installing Kubernetes cluster online")
            install_kubernetes_cluster_online()
        else:
            print("Installing Kubernetes cluster offline")
            install_kubernetes_cluster()
        check_coredns_schedule()


@Install.subcommand("operators")
class Operators(cli.Application):
    """install nfdgpu/netowrk/prometheus/mpi operator"""
    online = False
    software = cli.SwitchAttr("--software", str, mandatory=False, help="Install specified software, example: tensorboard,kruise")

    def main(self):
        self.parent._pass_online_flag(self)
        if self.online:
            print("Installing operators online")
            install_operators(self.software, "_online")
        else:
            print("Installing operators offline")
            # init_apt_source()
            install_operators(self.software)


@Install.subcommand("ceph-csi-cephfs")
class CephCsiCephfs(cli.Application):
    """install ceph csi cephfs"""

    def main(self):
        install_ceph_csi_cephfs()


@Deploy.subcommand("add")
class Add(cli.Application):
    """add node/object"""


@Add.subcommand("nodes")
class Nodes(cli.Application):
    """add kubernetes nodes"""

    def main(self):
        add_nodes()


@Deploy.subcommand("delete")
class Delete(cli.Application):
    """delete node/object"""


@Delete.subcommand("node")
class Node(cli.Application):
    """delete kubernetes node"""

    def main(self, node_name):
        delete_node(node_name)


@Deploy.subcommand("all")
class All(cli.Application):
    """deploy kubernetes cluster and install operators"""
    no_init_registry = cli.Flag("--no-init-registry", default=False, help="default: false")
    no_push_images = cli.Flag("--no-push-images", default=False, help="default: false")
    no_push_charts = cli.Flag("--no-push-charts", default=False, help="default: false")
    no_install_operators = cli.Flag("--no-install-operators", default=False, help="default: false")
    access_key = cli.SwitchAttr("--access-key", str, mandatory=True, help="cloud.nascentcore.ai user accessKey")
    api_address = cli.SwitchAttr("--api-address", str, mandatory=False, default="https://cloud.nascentcore.ai", help="api address")
    storage_class = cli.SwitchAttr("--storage-class", str, mandatory=False, default="ceph-filesystem", help="storageClass name")
    oss_bucket = cli.SwitchAttr("--oss-bucket", str, mandatory=False, default="sxwl-ai", help="upload model to oss bucket name")
    log_level = cli.SwitchAttr("--log-level", str, mandatory=False, default="INFO", help="log level")

    def main(self):
        if not self.no_init_registry:
            init_registry()

        if not self.no_push_images:
            create_project()
            push_images()

        install_kubernetes_cluster()
        init_cpod_info(self.access_key, self.api_address, self.storage_class, self.oss_bucket, self.log_level)
        check_coredns_schedule()

        if not self.no_push_charts:
            install_helm_push()
            push_charts()

        if not self.no_install_operators:
            init_apt_source()
            install_operators()

        print(colors.green | "===== [3kctl] deployment successful =====")
