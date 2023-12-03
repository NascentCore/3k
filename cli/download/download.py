#!/usr/bin/env python3

import math

from .model_scope import ModelScopeHub
from plumbum import cli, local, RETCODE, colors, TF


class Download(cli.Application):
    """Download model or dataset"""


@Download.subcommand("model")
class Model(cli.Application):
    """download model"""

    def main(self, hub_name, model_id):
        hub = hub_factory(hub_name)
        if hub is None:
            print("hub {0} is not supported".format(hub_name))
            return

        model_size = math.ceil(hub.size(model_id))
        print("model {0} size {1} GB".format(hub_name, model_size))

        # 创建PVC
        ret = local["kubectl"]("get", "node", "-n cpod")
        print(ret)

        # 开启下载Job


def hub_factory(hub_name):
    if hub_name == "modelscope":
        return ModelScopeHub()
    else:
        return None
