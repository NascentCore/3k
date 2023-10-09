# Based on https://goproxy.cn/
# You need to `source go_proxy.sh` for the change to take effect
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct
