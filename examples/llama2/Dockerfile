# LY worker4:
# Ubuntu 20.04.6 LTS (GNU/Linux 5.4.0-164-generic x86_64)
# NVIDIA-SMI 535.104.05    Driver Version: 535.104.05    CUDA Version: 12.2

# https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
LABEL maintainer="glen@nascentcore.ai"
LABEL version="0.1"
LABEL build_date="2023-10-13"

# https://hub.docker.com/r/pytorch/pytorch/tags
#FROM pytorch/pytorch:latest
#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Foundamental env and tools.
RUN apt-get update && apt-get install -y git wget && \
    apt-get install -y python3-pip

WORKDIR /llama2

# Set up the pip source for China.
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# And then upgrade pip.
RUN python3 -m pip install --no-cache-dir --upgrade pip

# TODO: All artifiacts for training/fine-tuning LLaMA 2.
COPY llama2_demo.py requirements.txt config \
     Llama2-Chinese-7b-Chat /llama2/

# Dependencies for Python app.
RUN python3 -m pip install -r requirements.txt

#ENTRYPOINT ["python3", "llama2_demo.py"]
