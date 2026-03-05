#!/bin/bash
# Calico CNI 兜底部署脚本
# 当 KubeKey 未能自动部署 Calico 时使用
# 使用阿里云镜像源，适配国内网络环境

CALICO_VERSION="${CALICO_VERSION:-v3.23}"
CALICO_IMAGE_VERSION="${CALICO_IMAGE_VERSION:-v3.23.2}"
REGISTRY="${REGISTRY:-registry.cn-beijing.aliyuncs.com/kubesphereio}"

# KubeKey 默认 pod CIDR
POD_CIDR="${POD_CIDR:-10.233.64.0/18}"

MANIFEST_URL="https://projectcalico.docs.tigera.io/archive/${CALICO_VERSION}/manifests/calico.yaml"
TMPFILE=$(mktemp /tmp/calico-XXXXXX.yaml)

echo "下载 Calico ${CALICO_VERSION} manifest..."
for i in $(seq 1 3); do
  if curl -fL -o "$TMPFILE" "$MANIFEST_URL"; then
    break
  fi
  echo "下载失败，重试 ($i/3)..."
  sleep 5
done

if [ ! -s "$TMPFILE" ]; then
  echo "错误: 无法下载 Calico manifest"
  exit 1
fi

echo "替换镜像为阿里云源..."
# 替换所有 docker.io/calico/* 镜像为阿里云镜像，并统一版本号
sed -i \
  -e "s|docker.io/calico/cni:[^ ]*|${REGISTRY}/cni:${CALICO_IMAGE_VERSION}|g" \
  -e "s|docker.io/calico/node:[^ ]*|${REGISTRY}/node:${CALICO_IMAGE_VERSION}|g" \
  -e "s|docker.io/calico/kube-controllers:[^ ]*|${REGISTRY}/kube-controllers:${CALICO_IMAGE_VERSION}|g" \
  -e "s|docker.io/calico/pod2daemon-flexvol:[^ ]*|${REGISTRY}/pod2daemon-flexvol:${CALICO_IMAGE_VERSION}|g" \
  "$TMPFILE"

echo "设置 Pod CIDR: ${POD_CIDR}..."
sed -i \
  -e 's|# - name: CALICO_IPV4POOL_CIDR|- name: CALICO_IPV4POOL_CIDR|' \
  -e "s|#   value: \"192.168.0.0/16\"|  value: \"${POD_CIDR}\"|" \
  "$TMPFILE"

echo "部署 Calico..."
kubectl apply -f "$TMPFILE"

echo "等待 Calico 就绪..."
kubectl -n kube-system rollout status daemonset/calico-node --timeout=120s
kubectl -n kube-system rollout status deployment/calico-kube-controllers --timeout=120s

rm -f "$TMPFILE"
echo "Calico 部署完成"
