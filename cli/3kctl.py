#!/usr/bin/env python3
from plumbum import cli, colors
from deploy.deploy import Deploy


class MainApp(cli.Application):
    PROGNAME = "cli" | colors.green
    VERSION = "0.1" | colors.blue
    DESCRIPTION = "CLI tool for 3k platform"


if __name__ == '__main__':
    MainApp.subcommand("deploy", Deploy)
    MainApp.run()
