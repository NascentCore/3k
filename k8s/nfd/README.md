# NFD

Node feature discovery deployment artifacts.

Use Helm to deploy https://github.com/kubernetes-sigs/node-feature-discovery

In order to be compatbile with NVIDIA GPU & Network operator, we also merge the nfd values.yaml file
from [GPU](https://github.com/NVIDIA/gpu-operator/blob/master/deployments/gpu-operator/values.yaml)
& [Network](https://github.com/Mellanox/network-operator/blob/master/deployment/network-operator/values.yaml)
operator and use them here.
