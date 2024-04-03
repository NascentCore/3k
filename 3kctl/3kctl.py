#!/opt/.venv/bin/python
from plumbum import cli, colors
from deploy.deploy import Deploy
from download.download import Download
from upload.upload import Upload
from serve.service import Serve


class MainApp(cli.Application):
    PROGNAME = "3kctl" | colors.green
    VERSION = "0.1" | colors.blue
    DESCRIPTION = "CLI tool for 3k platform"


if __name__ == '__main__':
    MainApp.subcommand("deploy", Deploy)
    MainApp.subcommand("download", Download)
    MainApp.subcommand("upload", Upload)
    MainApp.subcommand("serve", Serve)
    MainApp.run()
