#!/bin/bash -x
# 该脚本用于构造image，并上传华为云镜像服务
# Usage: ./0.build_and_push_docker_image.sh
#-------------------------------------------------------------------

# 这几句先构造base镜像上传，base镜像不会轻易改动
#docker rmi swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
#docker build . -f base.Dockerfile -t swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
#docker push swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base

# 在base镜像基础上，加上nccl的小demo
docker rmi -f swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
docker build -t swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test .
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
