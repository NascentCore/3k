#!/bin/bash -x
# 这几句用构造好的镜像，在k8s上构造容器

kubectl delete pod/for-nccl-test
sudo ctr -n k8s.io images rm swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test:latest
kubectl apply -f ./k8s_nccl_test.yaml
