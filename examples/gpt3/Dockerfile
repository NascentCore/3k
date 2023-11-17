FROM registry.ap-southeast-1.aliyuncs.com/sxwl-ai/cuda_base:2023-10-23
WORKDIR /workspace
COPY . .
RUN python3 -m pip install modelscope megatron_util jieba \
    -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
# https://stackoverflow.com/questions/77433096/notimplementederror-loading-a-dataset-cached-in-a-localfilesystem-is-not-suppor
RUN pip3 install -U datasets
