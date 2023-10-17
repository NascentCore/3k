#!/bin/bash -x

# 这几句先构造base镜像上传，base镜像不会轻易改动
docker rmi swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
docker build . -f base.Dockerfile -t swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
exit

docker rmi -f swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
docker build -t swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test .
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
