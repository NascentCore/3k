# NCCL

该repo为了构造用于测试NCCL连通性的镜像，包括裸机、容器上、k8s的容器上，竭力包含所有必要工具。

| 文件/文件夹 | 用途 |
| :-----| :----- |
|`README.md`|本文档|
|`nccl`、`nccl-tests`|nccl的源码和测试nccl性能的小工具。https://github.com/NVIDIA/nccl https://github.com/NVIDIA/nccl-tests|
|`0001-solve-compilation-error.patch`、`gpu_direct_rdma_access`|直接调用cuda函数，跑通GPUDirect RDMA；要求有nv_peer_memory内核模块。`0001-solve-compilation-error.patch`修正源码中的编译错误bug。 https://github.com/Mellanox/gpu_direct_rdma_access|
|`nccl_test_locally.py`|单机多卡情况下的allreduce测试用例|
|`dist_nccl_demo.py`|多机情况下的allreduce测试用例|
|`DDP_MNIST_demo.py`|分布式训练的例子；各结点独自训练的间隙，调用allreduce求梯度平均。调用allreduce的途径与`dist_nccl_demo.py`相同；`dist_nccl_demo.py`运行正常，此例应当运行正常。|
|`base.Dockerfile`、`Dockerfile`、`entrypoint.sh`、`0.build_and_push_docker_image.sh`|镜像本体的描述文件；和制作镜像的命令|
|`1.k8s_apply_yaml.sh`、`k8s_nccl_test.yaml`|一个例子，在k8s上运行该镜像|
|`2.docker_run.sh`|一个例子，在docker上运行该镜像|

其中的主要测试用例是调用NCCL做一次allreduce，包括单机多卡情况（`nccl_test_locally.py`）、和多机情况（`dist_nccl_demo.py`）。只要allreduce跑通，则可以在此配置下跑通分布式训练。

## 构造容器镜像（`0.build_and_push_docker_image.sh`）、在k8s下启动镜像（`1.k8s_apply_yaml.sh`）、在docker下启动镜像（`2.docker_run.sh`）

### 构造容器镜像（`0.build_and_push_docker_image.sh`）

#### 1、首先构造底包（`base.Dockerfile`）

底包中包含了特别大、绝对没有争议的驱动、环境，包括python3、MLNX_OFED用户部分程序、pytorch和其他常见网络命令。

```
docker rmi swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
docker build . -f base.Dockerfile -t swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base
```

#### 2、然后在底包的基础上加入各种工具（`Dockerfile`）

首先配好sshd，然后把该repo中的文件都复制进去。

```
docker rmi -f swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
docker build -t swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test .
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test
```

#### 3、镜像的`entrypoint.sh`

启动镜像后，会自动编译`gpu_direct_rdma_access`、`nccl`、`nccl_test`；然后跑nccl_test。这部分代码在`entrypoint.sh`中。
编译流程要求镜像中有cuda工具链，也就是说，host上的nvidia-container-toolkit运行正常；RuntimeClass是nvidia，把nvidia驱动和工具链都放在容器内。
运行流程，希望容器可以认到两块显卡，否则`nccl_test_locally.py`会跑错。

### 在k8s下启动镜像（`1.k8s_apply_yaml.sh`、`k8s_nccl_test.yaml`）

对k8s的要求主要是GPU operator就绪；而且可以申请到两块GPU。

### 在docker下启动镜像（`2.docker_run.sh`）

在k8s的情况下启动容器，容器中使用GPU和RDMA网卡的一切特殊参数均被包圆。在docker下，就要手动设置这些参数。
`2.docker_run.sh`是让容器认到GPU和RDMA网卡的例子，主要内容如下：

```
docker run -it --rm \
    --shm-size=1g \
    --runtime=nvidia \
    --name ${DOCKER_NAME} \
    --cap-add=IPC_LOCK \
    --network host \
    --device=/dev/infiniband/uverbs0 \
    --hostname ${DOCKER_NAME} \
    ${IMAGE_NAME}
```

其中，`--shm-size=1g`的依据是这里https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data；没有这个参数，NCCL走SHM途径传输数据失败。

`--runtime=nvidia`触发nvidia-container-toolkit向容器中注入nvidia驱动和cuda工具链；没有这个参数，nvidia-container-toolkit配置好的情况下也会自动触发。

`--cap-add=IPC_LOCK`是使用RDMA网卡必要的参数。

`--network host`将IPoIB模式的RDMA网卡、和Ethernet模式的RDMA网卡的ifname直通给容器内部。

`--device=/dev/infiniband/uverbs0`将作为PCIe设备的RDMA网卡直通给容器内部。

RDMA网卡的相关参数来自《How-to: Deploy RDMA accelerated Docker container over InfiniBand fabric. - Solutions - NVIDIA Networking Docs》这篇文章，原链接https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785，不知道为啥，这篇文章被删掉了。

## nccl_test_locally.py
This test file runs an all_reduce operation on equal number of tensors as the GPU count on localhost.
You can modify the reduce operation to test different operations.
```
python nccl_test_locally.py
```

## Building NCCL

[Instructions](https://github.com/NVIDIA/nccl#install)

```
cd nccl
make -j src.build

# Build deb package
sudo apt-get install devscripts
make pkg.debian.build

# Install libnccl2
sudo dpkg -i build/pkg/deb/libnccl2_2.18.6-1+cuda11.7_amd64.deb
sudo dpkg -i build/pkg/deb/libnccl-dev_2.18.6-1+cuda11.7_amd64.deb
```

## NCCL-TESTS

* An in-place operation uses the same buffer for its output as was used to
  provide its input. An out-of-place operation has distinct input and output
  buffers [ref](https://github.com/NVIDIA/nccl/issues/12).

## NCCL PyTorch tests

* Single GPU
  ```
  # Runs a single worker process on the localhost
  CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 MASTER_PORT=29500 MASTER_ADDR=127.0.0.1 python dist_nccl_demo.py
  ```
* Distributed
  Start N workers, can be on localhost or across multiple hosts, forms distributed collective communication:
  ```
  WORLD_SIZE=n RANK=0 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py
  WORLD_SIZE=n RANK=1 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py 
  WORLD_SIZE=n RANK=2 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py 
  ...
  WORLD_SIZE=n RANK=n-1 MASTER_PORT=29500 MASTER_ADDR=<MASTER-IP> ./dist_nccl_demo.py  
  ```

## NOTES

* NCCL always use chains [LL-128](https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990)
  for intra-node all reduce, see [NVIDIA/nccl/issues/919](https://github.com/NVIDIA/nccl/issues/919)

## 在worker4、worker5上做NCCL连通性测试（数据通过NET/Socket）

在worker4上运行
```
./go_worker4.sh
```

在worker5上运行
```
./go_worker5.sh
```
