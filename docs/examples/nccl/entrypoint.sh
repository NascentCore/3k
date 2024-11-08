#!/bin/bash -x
#
# The entrypoint of the 'for_nccl_test' image
#
# -------------------------------------------------------------------

nvidia-smi

cd gpu_direct_rdma_access/
patch -p1 < ../0001-solve-compilation-error.patch
make USE_CUDA=1
./server -a 192.168.0.206 -n 10000 -D 1 -s 10000000 -p 18001 &
cd ..

ifconfig -a
ibv_devinfo
ibdev2netdev -v
ip link show

python ./nccl_test_locally.py

# 编译nccl库
cd nccl

# 如果是为了使用，这么make
make -j src.build

# 如果是为了源码级调试，这么make带debug信息。
# NVCC_GENCODE变量的值会传入cuda编译器nvcc
# 如果不特地指定的话，默认情况下会为所有GPU微结构生成代码，带debug信息的代码体积很大，导致link失败。
# 不同显卡的sm号从这里查，https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# make -j src.build TRACE=1 VERBOSE=1 DEBUG=1 NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80"

cd ..

# 编译nccl-tests
cd nccl-tests

# 修改Makefile里的link参数。
# 如果不改，运行时不管动态库搜索路径的话，会动态链接到系统自带的nccl动态库/usr/lib/x86_64-linux-gnu/libnccl.so。
# 这里改成-lnccl_static，编译时静态链接到刚才编译好的nccl库，libnccl_static.a
sed -i "s/^LIBRARIES += nccl$/LIBRARIES += nccl_static/" ./src/Makefile

make -j NCCL_HOME=$(pwd)/../nccl/build

# 简单跑跑试试

go() {
# -b 512M -e 512M 意思是集合运算的数据量，两个512M数量相同
# -g 2 指的是两块GPU，这个和CUDA_VISIBLE_DEVICE环境变量对应
CUDA_VISIBLE_DEVICE=0,1 \
    ./build/all_reduce_perf \
    -b 512M -e 512M \
    -g 2
}

echo P2P
export NCCL_P2P_LEVEL=SYS
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_SHM_DISABLE=1
go

echo NET/IB
export NCCL_P2P_LEVEL=LOC
export NCCL_NET=IB
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_SHM_DISABLE=1
go

echo SHM
export NCCL_P2P_LEVEL=LOC
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_SHM_DISABLE=0
go

echo NET/Socket
export NCCL_P2P_LEVEL=LOC
export NCCL_NET=Socket
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_SHM_DISABLE=1
go

sleep inf
