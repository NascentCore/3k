#!/usr/bin/env python3
from plumbum import cli
from .registry import *
from .kubernetes import *

class DeployCluster(cli.Application):
    """cluster offline deployment"""
    _all = False
    _init_registry = _push_images = _create_cluster = _install_apps = False
    _install_ceph = _add_nodes = False
    _node = ""

    @cli.autoswitch()
    def init_registry(self):
        """initialize the private repository"""
        self._init_registry = True

    @cli.autoswitch()
    def push_images(self):
        """push images to private repository"""
        self._push_images = True

    @cli.autoswitch()
    def create_cluster(self):
        """deploy kubernetes cluster"""
        self._create_cluster = True

    @cli.autoswitch()
    def install_apps(self):
        """install nfd/gpu/network/mpi operator"""
        self._install_apps = True

    @cli.autoswitch()
    def install_ceph(self):
        """install ceph cluster"""
        self._install_ceph = True

    @cli.autoswitch()
    def add_nodes(self):
        """add nodes for kubernetes cluster"""
        self._add_nodes = True

    @cli.autoswitch(str)
    def delete_node(self, args):
        """delete node for kubernetes cluster"""
        self._node = args

    @cli.switch(["-a", "--all"])
    def all(self):
        """deploy kubernetes cluster and install apps"""
        self._all = True

    def main(self):
        if self._all:
            init_registry()
            create_project()
            push_images()
            create_cluster()
            install_helm_push()
            push_charts()
            install_apps()
            print(colors.green | "===== [3kctl] deployment successful =====")

        else:
            if self._init_registry:
                init_registry()

            if self._push_images:
                create_project()
                push_images()

            if self._create_cluster:
                create_cluster()

            if self._install_apps:
                install_helm_push()
                push_charts()
                install_apps()

            if self._install_ceph:
                install_ceph()

            if self._add_nodes:
                add_nodes()

            if self._node:
                delete_node(self._node)
