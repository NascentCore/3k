#!/bin/bash

# 三台机器都执行
# 重置kubeadm，kubekey底层是用kubeadm安装的
kubeadm reset
# 清理reset残余文件
# /etc/cni/net.d 是 CNI 网络插件相关的残余文件
rm -rf /etc/cni/net.d   
# 清除全部的iptable中的规则
iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X
# TODO: 需要调研ipvsadm作用
ipvsadm --clear
# 删除kube的配置文件
rm -f $HOME/.kube/config

# 三台机器都执行，删除k8s相关的目录和组件
sudo apt-get purge kubeadm kubectl kubelet kubernetes-cni kube*
sudo apt autoremove
rm -rf /etc/systemd/system/kubelet.service
rm -rf /etc/systemd/system/kube*
sudo rm -rf ~/.kube
sudo rm -rf /etc/kubernetes/
sudo rm -rf /var/lib/kube*
rm -rf /var/lib/containerd

# stop容器运行时并disable
sudo systemctl stop docker.service
sudo systemctl disable docker.service
sudo systemctl stop containerd.service
sudo systemctl disable containerd.service
sudo apt-get purge docker-ce docker-ce-cli containerd.io
sudo apt-get remove docker docker-engine docker.io containerd runc

# 删除docker相关目录和文件
sudo rm -rf /var/lib/docker /var/lib/containerd /etc/docker
sudo rm /usr/bin/docker
sudo rm /usr/bin/containerd
# 删除contained命令及配置
rm -rf /usr/local/bin/
rm -rf /etc/containerd/
# 删除containerd服务
rm -rf /usr/local/lib/systemd/system/containerd.service
# 删除runc
rm -rf /usr/local/sbin/runc
# 删除CNI插件
rm -rf /opt/containerd/
# 删除ctr命令
rm -rf /usr/bin/ctr

# 自动删除不必要的apt包
sudo apt autoremove
