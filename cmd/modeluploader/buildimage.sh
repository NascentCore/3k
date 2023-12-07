#!/bin/bash
CGO_ENABLED=0  GOOS=linux  GOARCH=amd64  go build modeluploadjob.go
docker build -f ./Dockerfile . -t registry.cn-hangzhou.aliyuncs.com/sxwl-ai/modeluploader:`git rev-parse --short HEAD`
