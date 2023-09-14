#!/bin/bash -x

kubectl delete pod/for-nccl-test
kubectl apply -f ./k8s_nccl_test.yaml
