FROM swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base:latest
WORKDIR /workspace
COPY LLaMA-Factory ./
#COPY acclerate_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml
COPY dataset_info.json /data/dataset/
RUN pip3 install --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple  -r requirements.txt  deepspeed  -U transformers
RUN pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple  tensorboard