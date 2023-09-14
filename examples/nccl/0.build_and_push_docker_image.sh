#!/bin/bash -x

#docker build . -f base.Dockerfile -t swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
#docker push swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
docker build -t swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test .
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
