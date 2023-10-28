#!/usr/bin/env python3
import os
import sys
import yaml
from yaml import SafeLoader
from plumbum import cli, local, RETCODE, colors, TF
from plumbum.cmd import ls, scp


class Dict(dict):
    __setattr__ = dict.__setattr__
    __getattr__ = dict.__getitem__


def dict_to_obj(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    d = Dict()
    for k, v in dict_obj.items():
        d[k] = dict_to_obj(v)
    return d


def parse_ini():
    """load configuration"""
    conf = cli.Config("conf/config.ini")
    conf.read()
    c = dict_to_obj(conf.parser._sections)
    if not c:
        print(colors.red | "the configuration is empty, please check the configuration file.")
        sys.exit(1)

    return c


def parse_yaml():
    with open("conf/config-sample.yaml", "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    y = dict_to_obj(data)

    return y


def kk_run(action, obj, *args):
    kk = local[os.path.join(Conf.c.deploy.work_dir, Conf.c.deploy.kk_bin)]

    param = [action, obj, *args, "-f", Conf.c.deploy.cluster_config]

    if action != "delete":
        param.append("-a")
        param.append(Conf.c.deploy.package)

    retcode = kk[param] & RETCODE(FG=True)

    if retcode != 0:
        sys.exit(retcode)


def helm_run(*args):
    helm = local["helm"]
    retcode = helm[args] & RETCODE(FG=True)

    if retcode != 0:
        sys.exit(retcode)


def kubectl_run(*args):
    kubectl = local["kubectl"]
    retcode = kubectl[args] & RETCODE(FG=True)

    if retcode != 0:
        sys.exit(retcode)


def is_true(cmd, timeout=120):
    second = 0
    while True:
        retcode = cmd & RETCODE
        if retcode == 0:
            return True

        if second >= timeout:
            return False

        second += 0


def copy_package(node):
    for file in ls("package").split():
        if ".deb" in file:
            scp[
                "-o",
                "stricthostkeychecking=no",
                "-p",
                "{}".format(node["port"]),
                "package/{}".format(file),
                "{}@{}:/tmp".format(node["user"], node["internalAddress"])
            ] & TF(0)


class Conf:
    c = parse_ini()
    y = parse_yaml()
