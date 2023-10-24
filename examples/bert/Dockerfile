# cuda_base was built from base.Dockerfile
# You must change this to the right one you built from base.Dockerfile
FROM cuda_base
WORKDIR /workspace
RUN wget https://sxwl-ai.oss-cn-beijing.aliyuncs.com/hf-datasets/hf_wikitext.tar.gz && \
    mkdir dataset && tar zxvf hf_wikitext.tar.gz --directory dataset && \
    rm hf_wikitext.tar.gz
COPY *.py ./
COPY model_roberta_base ./model_roberta_base
