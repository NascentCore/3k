#!/usr/bin/env python3
from plumbum import cli, colors
from deploy.deploy import DeployCluster


class MainApp(cli.Application):
    PROGNAME = "cli" | colors.green
    VERSION = "0.1" | colors.blue
    DESCRIPTION = "CLI tool for 3k platform"

    def main(self, *args):
        pass


@MainApp.subcommand("deploy")
class DeployCluster(DeployCluster):
    pass


if __name__ == '__main__':
    MainApp.run()
