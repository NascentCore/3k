#!/usr/bin/env python3
import sys
from plumbum import colors, local, RETCODE, TF
from plumbum.cmd import grep, ssh, awk, ls, scp
from .utils import kk_run, Conf, helm_run, kubectl_run, is_true


def install_kubernetes_cluster():
    print(colors.yellow | "===== [3kctl] install depend packages for all nodes =====")

    local["dpkg"]["-i", "package/sshpass.deb"] & TF(0)
    for node in Conf.y.spec.hosts:
        retcode = install_depend_packages(node)
        if retcode > 0:
            print("failed: {}".format(node["name"]))
            sys.exit(retcode)
        print("success: {}".format(node["name"]))

    print(colors.yellow | "===== [3kctl] create kubernetes cluster =====")

    kk_run("create",
           "cluster",
           "-y"
           )


def install_operators():
    for app in Conf.c.deploy.helm_apps.split(","):
        print(colors.yellow | "===== [3kctl] install {} =====".format(app))

        helm_run("install",
                 app,
                 "harbor/{}".format(app),
                 "-n",
                 app,
                 "--create-namespace",
                 "-f",
                 "{}/deploy/values/{}.yaml".format(Conf.c.deploy.work_dir, app)
                 )

    for app in Conf.c.deploy.yaml_apps.split(","):
        print(colors.yellow | "===== [3kctl] install {} =====".format(app))

        kubectl_run("apply",
                    "-f",
                    "{}/deploy/yaml_apps/{}.yaml".format(Conf.c.deploy.work_dir, app)
                    )


def install_ceph():
    print(colors.yellow | "===== [3kctl] install rook-ceph & ceph-cluster =====")

    for node in Conf.c.deploy.ceph_nodes.split(","):
        retcode = (local["kubectl"]["get", "node", node, "--show-labels=true"] | grep["role=ceph"]) & RETCODE
        if retcode != 0:
            kubectl_run("label", "node", node, "role=ceph")

    helm_run("install",
             "rook-ceph",
             "harbor/rook-ceph",
             "-n",
             "rook-ceph",
             "--create-namespace",
             "-f",
             "{}/deploy/values/rook-ceph.yaml".format(Conf.c.deploy.work_dir)
             )

    if is_true(local["kubectl"]["get", "pods", "-n", "rook-ceph"]
               | grep["rook-ceph-operator"]
               | grep["Running"]
               | grep["1/1"]):
        helm_run("install",
                 "rook-ceph-cluster",
                 "harbor/rook-ceph-cluster",
                 "-n",
                 "rook-ceph",
                 "--create-namespace",
                 "-f",
                 "{}/deploy/values/rook-ceph-cluster.yaml".format(Conf.c.deploy.work_dir)
                 )
    else:
        print(colors.red | "===== [3kctl] rook-ceph-operator not ready, timeout: 120s =====")
        sys.exit(3)


def add_nodes():
    print(colors.yellow | "===== [3kctl] add nodes for kubernetes cluster =====")

    # 获取当前集群中的节点
    ret = (local["kubectl"]["get", "nodes"] | grep["-v", "STATUS"] | awk["{print $1}"])()
    cur_nodes = ret.split()

    # 遍历 yaml 文件中的 hosts，对新增节点安装必要的依赖包
    for node in Conf.y.spec.hosts:
        if node["name"] in cur_nodes:
            continue

        install_depend_packages(node)

    kk_run("add", "nodes", "-y")


def delete_node(node):
    print(colors.yellow | "===== [3kctl] delete node for kubernetes cluster =====")

    kk_run("delete", "node", node, "-y")


def install_depend_packages(node):
    pass_info, key_info = get_ssh_info(node)
    copy_package(node, pass_info, key_info)
    args = [
        "-o",
        "stricthostkeychecking=no",
        *key_info,
        "-p",
        "{}".format(node["port"]),
        "{}@{}".format(node["user"], node["internalAddress"]),
    ]
    cmd1 = ["sudo", "dpkg", "-i", "/tmp/*.deb"]
    cmd2 = ["sudo", "rm", "-fv", "/tmp/*.deb"]

    retcode = 0
    for cmd in [cmd1, cmd2]:
        if pass_info:
            retcode += local["sshpass"][pass_info + ["ssh"] + args + cmd] & RETCODE
        else:
            retcode += ssh[args + cmd] & RETCODE

    return retcode

def copy_package(node, pass_info, key_info):
    for file in ls("package").split():
        if ".deb" in file:
            args = [
                "-o",
                "stricthostkeychecking=no",
                *key_info,
                "-P",
                "{}".format(node["port"]),
                "package/{}".format(file),
                "{}@{}:/tmp".format(node["user"], node["internalAddress"])
            ]
            if pass_info:
                local["sshpass"][pass_info + ["scp"] + args] & TF(0)
            else:
                scp[args] & TF(0)


def get_ssh_info(node):
    pass_info = key_info = []
    if "password" in node:
        pass_info = ["-p", node["password"]]

    if "privateKeyPath" in node:
        key_info = ["-i", node["privateKeyPath"]]

    return pass_info, key_info
