# NFD

Values.yaml file for installing [Node Feature Discovery](
https://github.com/kubernetes-sigs/node-feature-discovery) (NFD) components with
HELM.

In order to be compatible with NVIDIA GPU & Network operator, we also merge the
nfd values.yaml file from
[GPU](https://github.com/NVIDIA/gpu-operator/blob/master/deployments/gpu-operator/values.yaml)
[Network](https://github.com/Mellanox/network-operator/blob/master/deployment/network-operator/values.yaml)
operator and use them here.

Primarily the `deviceClassWhiteList` config is changed to include all of the GPU
and InfiniBand devices.
