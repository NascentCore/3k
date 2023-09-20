#!/bin/bash -x

kubectl delete pod/for-nccl-test
sudo ctr -n k8s.io images rm swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test:latest
kubectl apply -f ./k8s_nccl_test.yaml
