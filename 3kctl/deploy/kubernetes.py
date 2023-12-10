#!/usr/bin/env python3
import sys
import time
import hashlib
from plumbum import colors, local, RETCODE, TF
from plumbum.cmd import grep, ssh, awk, ls, scp, sed
from .utils import kk_run, Conf, helm_run, kubectl_run, is_true


def install_kubernetes_cluster():
    print(colors.yellow | "===== [3kctl] install depend packages for all nodes =====")

    local["dpkg"]["-i", "packages/sshpass.deb"] & TF(0)
    for node in Conf.y.spec.hosts:
        retcode = install_dependent_packages(node)
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

    for node in Conf.c.deploy.ib_nodes.split(","):
        retcode = (local["kubectl"]["get", "node", node, "--show-labels=true"] | grep["infiniband=available"]) & RETCODE
        if retcode != 0:
            kubectl_run("label", "node", node, "infiniband=available")

    for app in Conf.c.deploy.yaml_apps.split(","):
        print(colors.yellow | "===== [3kctl] install {} =====".format(app))

        kubectl_run("create",
                    "-f",
                    "{}/deploy/yaml_apps/{}.yaml".format(Conf.c.deploy.work_dir, app)
                    )


def install_ceph_csi_cephfs():
    print(colors.yellow | "===== [3kctl] install ceph-csi-cephfs =====")

    retcode = grep["-E", "<cluserID>|<adminKey>", "{}/deploy/values/ceph-csi-cephfs.yaml".format(Conf.c.deploy.work_dir)] & RETCODE
    if retcode != 0:
        print(colors.red | "===== [3kctl] ceph clusterID or adminKey not configure, please check 'deploy/values/ceph-csi-cephfs.yaml' =====")
        sys.exit(4)

    helm_run("install", "ceph-csi-cephfs",
                     "harbor/ceph-csi-cephfs",
                     "-n",
                     "ceph-csi-cephfs",
                     "--create-namespace",
                     "-f",
                     "{}/deploy/values/ceph-csi-cephfs.yaml".format(Conf.c.deploy.work_dir)
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

        install_dependent_packages(node)

    kk_run("add", "nodes", "-y")


def delete_node(node):
    print(colors.yellow | "===== [3kctl] delete node for kubernetes cluster =====")

    kk_run("delete", "node", node, "-y")


def install_dependent_packages(node):
    pass_info, key_info = get_ssh_info(node)
    copy_packages(node, pass_info, key_info)
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

def copy_packages(node, pass_info, key_info):
    for file in ls("packages").split():
        if ".deb" in file:
            args = [
                "-o",
                "stricthostkeychecking=no",
                *key_info,
                "-P",
                "{}".format(node["port"]),
                "packages/{}".format(file),
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


def gen_cpod_id():
    ns_uid = local["kubectl"]("get",
                              "ns",
                              "kube-system",
                              "-o=jsonpath={.metadata.uid}"
                              )

    return hashlib.md5(str.encode(ns_uid)).hexdigest()


def init_cpod_info(access_key):
    if not (local["kubectl"]["get", "ns"] | grep["^cpod"]) & TF(0):
        kubectl_run("create", "ns", "cpod")

    local["kubectl"]["delete", "cm", "cpod-info", "-n", "cpod"] & TF(0)

    kubectl_run(
        "create",
        "cm",
        "cpod-info",
        "-n",
        "cpod",
        "--from-literal=access_key={}".format(access_key),
        "--from-literal=cpod_id={}".format(gen_cpod_id())
    )


def get_coredns_pods():
    second = 0
    while second < 120:
        ret = (local["kubectl"][
                   "get",
                   "pods",
                   "-n",
                   "kube-system",
                   "-owide"
               ] | grep["coredns"] | grep["Running"])(retcode=None)
        pods = ret.strip().split("\n")

        if len(pods) == 2:
            return pods

        time.sleep(1)
        second += 1

    return None


def check_coredns_schedule():
    pods = get_coredns_pods()
    if not pods:
        print(colors.red | "===== [3kctl] coredns not ready, timeout: 120s =====")
        sys.exit(4)

    pod1 = pods[0].split()
    pod2 = pods[1].split()

    if pod1[6] == pod2[6]:
        local["kubectl"]["delete", "pod", pod1[0], "-n", "kube-system"] & TF(0)
        print(colors.yellow | "===== [3kctl] coredns reschedule success =====")


def init_apt_source():
    retcode = (local["docker"]["ps"] | grep["alpine-nginx"]) & RETCODE

    if retcode != 0:
        retcode = local["docker"][
            "run",
            "-tid",
            "--name",
            "apt-source",
            "-p",
            "8080:80",
            "-v",
            "{}/packages:/usr/share/nginx/html".format(Conf.c.deploy.work_dir),
            "dockerhub.kubekey.local/kubesphereio/alpine-nginx:main"
        ] & RETCODE

        if retcode != 0:
            print(colors.red | "===== [3kctl] start alpin-nginx failed =====")
            sys.exit(4)

    sed["-i", "s/master_ip/{}/".format(Conf.y.spec.hosts[0]["internalAddress"]), "packages/sources.list"] & TF(0)
    for namespace in ["gpu-operator", "network-operator"]:
        local["kubectl"][
            "create",
            "configmap",
            "repo-config",
            "-n",
            namespace,
            "--from-file=packages/sources.list"
        ] & TF(0)
