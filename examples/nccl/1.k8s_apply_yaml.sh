#!/bin/bash -x
# 该脚本新建一个k8s容器，用来做NCCL连通性测试
# bash ./1.k8s_apply_yaml.sh
#------------------------------------------------------------

kubectl delete pod/for-nccl-test
sudo ctr -n k8s.io images rm swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test:latest
kubectl apply -f ./k8s_nccl_test.yaml
