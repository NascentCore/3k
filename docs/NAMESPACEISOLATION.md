三千平台 Kubernetes 基于 namespace 隔离

### Kubernetes 中的 namespace 概念

Kubernetes 中的 namespace 是一种将资源隔离的方式，它将一个物理的集群分割成多个虚拟的集群，每个虚拟的集群称为一个 namespace。每个 namespace 都有自己独立的资源，如 Pod、Service、ConfigMap 等，这些资源只能在该 namespace 中被访问和管理。这样，多个团队或者项目就可以共享一个 Kubernetes 集群，而不会相互干扰。

### namespace 隔离的目的和好处

namespace 隔离的主要目的是为了提高 Kubernetes 集群的安全性、可管理性和资源利用率。通过将不同的应用程序、服务或用户分配到不同的 namespace 中，可以实现以下好处：

- 资源隔离：不同 namespace 中的资源（如 CPU、内存、存储等）是隔离的，一个 namespace 中的应用程序不能使用其他 namespace 中的资源，从而避免了资源竞争和冲突。
- 安全隔离：不同 namespace 中的应用程序之间是相互隔离的，它们不能直接访问其他 namespace 中的资源，从而提高了系统的安全性。
- 管理隔离：不同 namespace 中的应用程序可以由不同的团队或用户管理，从而实现了多租户管理和资源分配的灵活性。

### 三千平台 Kubernetes namespace 隔离的实现

1. 为每个用户初始化一个 namespace
   用户任务第一次被调度到三千平台时，为用户在 kubernetes 集群初始化 namespace，并且集群使用 public namespace 保存公开的数据集和模型。用户的推理训练任务如果使用公开的数据集和模型，会从 public namespace 讲模型和数据集拷贝到用户命名空间，这里拷贝只会拷贝 kubernetes 的资源对象（pv、pvc、modelstorage、datase），不会实际拷贝模型文件，用户对公开模型和数据集只有只读权限。
2. 配置资源限制
   为每个 namespace 配置资源限制，以确保每个 namespace 中的应用程序不会消耗过多的资源。可以使用 ResourceQuota 对象来配置资源限制，例如限制 namespace 中的 CPU、内存、存储等资源的使用量。
3. 网络隔离
   为每个 namespace 配置网络隔离，以确保不同 namespace 中的应用程序之间不会相互干扰。可以使用 NetworkPolicy 对象来配置网络隔离，例如限制 namespace 中的 Pod 之间的通信、限制外部网络对 namespace 中的 Pod 的访问等。
4. 安全策略
   为每个 namespace 配置安全策略，以确保只有授权的用户或应用程序可以访问 namespace 中的资源。可以使用 Role、RoleBinding、ServiceAccount 等对象来配置安全策略，例如限制对 namespace 中的资源的访问权限、配置身份验证和授权等。
