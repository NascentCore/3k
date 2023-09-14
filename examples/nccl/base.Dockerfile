FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
RUN apt update
RUN apt install -y python3-pip python-is-python3

# Change pip source to THU source, after changing pip source, use Python3 to
# reinstall pip，then use Python3 to invoke pip.
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip install torch
