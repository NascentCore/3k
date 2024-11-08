# cuda_base was built from base.Dockerfile
# You must change this to the right one you built from base.Dockerfile
FROM registry.ap-southeast-1.aliyuncs.com/sxwl-ai/cuda_base:2023-10-23
WORKDIR /workspace
# Include dataset for hermetic testing
RUN wget --progress=dot:giga https://sxwl-ai.oss-cn-beijing.aliyuncs.com/hf-datasets/hf_wikitext.tar.gz && \
    mkdir dataset && tar zxvf hf_wikitext.tar.gz --directory dataset && \
    rm hf_wikitext.tar.gz
# Allow editing source file when exec inside the container
RUN apt-get update --fix-missing && apt-get install -y vim --fix-missing
COPY *.py ./
COPY model_roberta_base ./model_roberta_base
