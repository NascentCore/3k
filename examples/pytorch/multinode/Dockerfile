FROM registry.ap-southeast-1.aliyuncs.com/sxwl-ai/cuda_base:2023-10-23
WORKDIR /workspace
RUN pip3  install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
COPY main.py datautils.py ./
