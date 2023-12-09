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
    access_key = cli.SwitchAttr("--access-key", str, mandatory=True,
                                help="cloud.nascentcore.ai user accessKey")

    def main(self):
        init_cpod_info(self.access_key)


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


@Install.subcommand("kubernetes")
class Kubernetes(cli.Application):
    """install kubernetes cluster"""

    def main(self):
        install_kubernetes_cluster()
        check_coredns_schedule()


@Install.subcommand("operators")
class Operators(cli.Application):
    """install nfdgpu/netowrk/prometheus/mpi operator"""

    def main(self):
        init_apt_source()
        install_operators()


@Install.subcommand("ceph")
class Ceph(cli.Application):
    """install rook ceph cluster"""

    def main(self):
        install_ceph()


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
    access_key = cli.SwitchAttr("--access-key", str, mandatory=True,
                                help="cloud.nascentcore.ai user accessKey")

    def main(self):
        if not self.no_init_registry:
            init_registry()

        if not self.no_push_images:
            create_project()
            push_images()

        install_kubernetes_cluster()
        init_cpod_info(self.access_key)
        check_coredns_schedule()

        if not self.no_push_charts:
            install_helm_push()
            push_charts()

        if not self.no_install_operators:
            init_apt_source()
            install_operators()

        print(colors.green | "===== [3kctl] deployment successful =====")
