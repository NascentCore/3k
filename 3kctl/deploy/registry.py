#!/usr/bin/env python3
import sys
from plumbum import colors, local, TF, RETCODE
from plumbum.cmd import curl, mkdir, tar, ls
from .utils import kk_run, Conf, helm_run


def init_registry():
    print(colors.yellow | "===== [3kctl] init registry =====")

    local["systemctl"]["restart", "containerd"] & TF(0)
    local["systemctl"]["restart", "docker"] & TF(0)
    kk_run("init", "registry")


def push_images():
    print(colors.yellow | "===== [3kctl] push images to registry =====")

    kk_run("artifact", "image", "push")


def create_project():
    print(colors.yellow | "===== [3kctl] create registry project =====")
    retcode = curl[
                   "-u",
                   "{}:{}".format(Conf.c.registry.harbor_user, Conf.c.registry.harbor_pass),
                   "-X",
                   "POST",
                   "-H",
                   "Content-Type: application/json",
                   "{}/api/v2.0/projects".format(Conf.c.registry.harbor_addr),
                   "-d",
                   '{"project_name": "kubesphereio", "public": true}', "-k"
              ] & RETCODE(FG=True)
    if retcode != 0:
        sys.exit(retcode)


def install_helm_push():
    print(colors.yellow | "===== [3kctl] install helm-push plugin =====")

    res = local["helm"]("env").split("HELM_PLUGINS=")[1].split()[0].strip('"')
    plugin_dir = "{}/helm-push".format(res)
    mkdir("-p", plugin_dir)
    tar("xvf",
        "{}/helm-plugins/helm-push_0.10.4_linux_amd64.tar.gz".format(Conf.c.deploy.work_dir),
        "-C",
        plugin_dir)

    if "cm-push" not in local["helm"]("plugin", "list"):
        print(colors.red | "===== [3kctl] helm-push plugin install failed =====")
        sys.exit(2)


def push_charts():
    print(colors.yellow | "===== [3kctl] push helm charts to registry =====")

    if not local["helm"]["repo", "list"] & TF(0) or "harbor" not in local["helm"]("repo", "list"):
        helm_run("repo",
                 "add",
                 "harbor",
                 "{}/chartrepo/kubesphereio".format(Conf.c.registry.harbor_addr),
                 "--ca-file", Conf.c.registry.ca_file,
                 "--cert-file", Conf.c.registry.cert_file,
                 "--key-file", Conf.c.registry.key_file,
                 "--ca-file", Conf.c.registry.ca_file,
                 "--username={}".format(Conf.c.registry.harbor_user),
                 "--password={}".format(Conf.c.registry.harbor_pass)
                 )

    for chart in ls("{}/helm-charts".format(Conf.c.deploy.work_dir)).split():
        helm_run("cm-push",
                 "helm-charts/{}".format(chart),
                 "harbor",
                 "--ca-file", Conf.c.registry.ca_file)

    helm_run("repo", "update")
