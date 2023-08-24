- 默认宿主机安装了python，kubectl
- 下载三个文件任意文件夹，在master上运行build_k8s, 使用kubekey安装单节点集群:
  - chmod a+x build_k8s
  - ./build_k8s 1.24.3
- 当需要添加节点时：
  - 在node上安装依赖：
  - sudo apt-get install socat conntrack ebtables ipset
  - 在master上运行add_nodes,格式如下：
  - chmod a+x add_nodes
  - ./add_nodes node_name username password ip_address
  - node name 可以自己定义！
- 搭建完成后，使用 sudo -i 可以操作k8s集群


- 彻底删除集群可执行`remove_all.sh`脚本
