FROM swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base:latest
WORKDIR /workspace
COPY LLaMA-Factory ./
COPY dataset_info.json /data/dataset/
RUN pip3 install --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt