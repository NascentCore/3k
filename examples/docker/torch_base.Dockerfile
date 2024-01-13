# Matches the host's driver
# NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install python3 pip torch tzdata git wget
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install -y tzdata && \
    apt-get install -y python3-pip python-is-python3 git wget && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install torch==2.0.1

# Install OFED net-tools opensm iputils-ping
RUN wget "https://content.mellanox.com/ofed/MLNX_OFED-23.07-0.5.1.2/MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu20.04-x86_64.tgz" && \
    tar -xf MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu20.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu20.04-x86_64 && ./mlnxofedinstall --force --upstream-libs --dpdk && \
    apt install -y net-tools iproute2 opensm iputils-ping
