#!/usr/bin/env python3
import sys
import time
import hashlib
import subprocess
from plumbum import colors, local, RETCODE, TF
from plumbum.cmd import grep, ssh, awk, ls, scp, sed
from .utils import kk_run, Conf, helm_run, kubectl_run, is_true
from .software import load_installed_softwares, install_software


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


def install_operators(software=""):
    print(colors.yellow | "===== [3kctl] install softwares =====")

    installed_softwares = load_installed_softwares()
    for sw in Conf.s.softwares:
        if software and software.strip() != sw["name"]:
            continue
        install_software(sw, installed_softwares, Conf.s.softwares)


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
    # subprocess.run(["sudo", "apt-get", "install", "-y", "socat", "conntrack", "ebtables", "ipset", "ipvsadm"])

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


def init_cpod_info(access_key, api_address, storage_class, oss_bucket, log_level):
    if not (local["kubectl"]["get", "ns"] | grep["^cpod$"]) & TF(0):
        kubectl_run("create", "ns", "cpod")
    if not (local["kubectl"]["get", "ns"] | grep["^cpod-system"]) & TF(0):
        kubectl_run("create", "ns", "cpod-system")

    local["kubectl"]["delete", "cm", "cpod-info", "-n", "cpod-system"] & TF(0)

    kubectl_run(
        "create",
        "cm",
        "cpod-info",
        "-n",
        "cpod-system",
        f"--from-literal=access_key={access_key}",
        f"--from-literal=cpod_id={gen_cpod_id()}",
        f"--from-literal=api_address={api_address}",
        f"--from-literal=storage_class={storage_class}",
        f"--from-literal=oss_bucket={oss_bucket}",
        f"--from-literal=log_level={log_level}",
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
            f"{Conf.c.deploy.work_dir}/packages/apt-source:/usr/share/nginx/html",
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
