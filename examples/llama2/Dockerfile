# LY worker4:
# Ubuntu 20.04.6 LTS (GNU/Linux 5.4.0-164-generic x86_64)
# NVIDIA-SMI 535.104.05    Driver Version: 535.104.05    CUDA Version: 12.2

# https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

LABEL sxwl.ai.maintainer="glen@nascentcore.ai"
LABEL sxwl.ai.build-date="2023-10-13"
LABEL version="0.1"

# https://hub.docker.com/r/pytorch/pytorch/tags
#FROM pytorch/pytorch:latest
#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Foundamental env and tools.
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc python3-dev && \
    apt-get install --no-install-recommends -y git wget python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /llama2

# TODO: All artifiacts for training/fine-tuning LLaMA 2.
COPY llama2_demo.py requirements.txt Llama2-Chinese-7b-Chat \
     config wikitext-103-raw data llama2_output llama2_ckpt /llama2/

# Set up the pip source for China, upgrade pip, and then install deps.
# https://stackoverflow.com/a/72551258
ENV PIP_ROOT_USER_ACTION=ignore
# hadolint ignore=DL3013
RUN python3 -m pip config \
    set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

#ENTRYPOINT ["python3", "llama2_demo.py"]
