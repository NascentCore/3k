FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && \
    apt-get install -y build-essential

WORKDIR /basic_transformer

COPY . ./

ENV PIP_ROOT_USER_ACTION=ignore

RUN python3 -m pip config \
    set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt