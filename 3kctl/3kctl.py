#!/usr/bin/env python3
from plumbum import cli, colors
from deploy.deploy import Deploy
from download.download import Download


class MainApp(cli.Application):
    PROGNAME = "3kctl" | colors.green
    VERSION = "0.1" | colors.blue
    DESCRIPTION = "CLI tool for 3k platform"


if __name__ == '__main__':
    MainApp.subcommand("deploy", Deploy)
    MainApp.subcommand("download", Download)
    MainApp.run()
