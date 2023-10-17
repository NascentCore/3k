FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list

RUN apt update

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

RUN apt install -y python3-pip python-is-python3

# aria2 is a substitute for wget
RUN apt install -y aria2

# Change pip source to China mainland source, after changing pip source, use Python3 to
# reinstall pip，then use Python3 to invoke pip.
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip install torch==2.0.1

# 因为OFED安装脚本是用perl写的
RUN apt install -y perl
RUN aria2c "https://content.mellanox.com/ofed/MLNX_OFED-23.07-0.5.1.2/MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu20.04-x86_64.tgz"
RUN tar -xf MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu20.04-x86_64.tgz
RUN cd MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu20.04-x86_64 && ./mlnxofedinstall --force --upstream-libs --dpdk


RUN apt install -y net-tools
RUN apt install -y iproute2
RUN apt install -y opensm
RUN apt install -y iputils-ping
RUN apt install -y git
RUN apt install -y vim

