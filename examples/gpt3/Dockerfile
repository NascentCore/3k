FROM registry.ap-southeast-1.aliyuncs.com/sxwl-ai/cuda_base:2023-10-23
WORKDIR /workspace
COPY nlp_gpt3_text-generation_1.3B ./nlp_gpt3_text-generation_1.3B
COPY finetune_dureader.py finetune_poetry.py ./
RUN python3 -m pip install modelscope megatron_util jieba \
    -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
