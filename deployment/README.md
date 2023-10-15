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
- kubeconfig 位于 ～/.kube/config，如有需要将该文件拷贝到所需的位置
- 安装 kubeflow training operator
  ```
  git clone git@github.com:kubeflow/training-operator.git
  kubectl apply -k ./kubeflow/training-operator/manifests/overlays/standalone
  # 验证 Training Operator 注册了相应的 CRD
  kubectl get crd | grep pytorchjobs
  kubectl get pods -n kubeflow
  ```
- 删除集群
  ```
  # 首先生成集群的配置文件，包含集群相关的各类信息，指定了如何卸载；这里会生成 sample.yaml
  ./kk create config --from-cluster --filename=cluster.yaml
  # 然后使用这个配置文件删除整个集群
  ./kk delete cluster --filename=cluster.yaml
  ```
