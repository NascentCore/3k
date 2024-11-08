# IB

InfiniBand verifications

## ib\_{read,write}\_bw

```bash
ssh -f <remote_ip> ib_read_bw
ib_read_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

ssh -f <remote_ip> ib_write_bw
ib_write_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

# RDMA outlets
mst status -v
```

## ibping

Node A

```bash
ibstat
从ibstat的输出中任意选择一个CA以其Port，记录其Base lid

CA和Port是上面选出的
ibping -S -C <CA> -P <Port>
```

Node B

```bash
ibping -L <前面记录的lid>
```

注：以上命令皆需 root 用户执行（或 sudo）

## 测试集群所有 IB 节点通信

测试集群 IB 网卡通信是否正常，原理运行一个 elastic pytorchjob，该任务就是一个集合通信 all_reduce 调用，在每个节点
上初始化一个标量 1，然后通过 all_reduce SUM 方法对所有节点标量 1 求和，打印一个等于节点数 n 的标量。通过 pod 之间的
反亲和性设置，保证一个 ib 节点只运行一个 worker。如果集群所有带有 IB 网卡配置的节点能够正常通信，那么该 pytorchjob 将会
变为`Succeeded`状态。

编译镜像`docker build -t . registry.cn-beijing.aliyuncs.com/sxwl-ai/ib-check:latest`

获取 IB 节点数量

```bash
IBNODE=$(kubectl get nodes --selector=feature.node.kubernetes.io/rdma.capable="true"
| awk 'NR>1 {count++} END {print count}')
```

将 ib-check.yaml 中 IBNODE 替换并运行`kubectl create -f ib-check.yaml`

观察校验任务 ib-check pytorchjob 的状态是否为`Succeeded`.

```bash
kubectl get pytorchjob
```
