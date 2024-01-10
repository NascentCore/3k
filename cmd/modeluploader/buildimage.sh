#!/bin/bash
CGO_ENABLED=0  GOOS=linux  GOARCH=amd64  go build modeluploadjob.go
docker build -f ./Dockerfile . -t sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/modeluploader:$(git rev-parse --short HEAD)
docker push sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/modeluploader:$(git rev-parse --short HEAD)
