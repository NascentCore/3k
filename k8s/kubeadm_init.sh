#!/bin/bash -x
#
# An example of create a kube cluster by using kubeadm
#
# Usage: ./kubeadm_init.sh
# -------------------------------------------------------------------

systemctl restart containerd

kubeadm init \
        --kubernetes-version=v1.24.12 \
        --pod-network-cidr=192.168.0.0/24 \
        --cri-socket=unix:///run/containerd/containerd.sock \
        --image-repository registry.cn-hangzhou.aliyuncs.com/google_containers \
        --v=5

export KUBECONFIG=/etc/kubernetes/admin.conf

# Make the single master node allowed to be schedule as a worker
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl taint nodes --all node-role.kubernetes.io/control-plane-

# If failed to apply the yaml file online, manually download it and apply the downloaded local yaml fail.
# wget https://docs.projectcalico.org/manifests/calico.yaml
# kubectl apply -f calico.yaml
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# Same as above
# wget https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
# kubectl create -f nvidia-device-plugin.yml
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Install the GPU Operator
kubectl apply -k node-feature-discovery/deployment/overlays/default
helm install --wait --generate-name \
     -n nvidia-gpu-operator --create-namespace \
      nvidia-gpu-operator/gpu-operator \
      --set operator.cleanupCRD=true \
      --set psp.enabled=false \
      --set operator.defaultRuntime=containerd \
      --set nfd.enabled=false \
      --set driver.enable=false \
      --set toolkit.enabled=false
