FROM registry.ap-southeast-1.aliyuncs.com/sxwl-ai/cuda_base:2023-10-23
RUN python3 -m pip install modelscope megatron_util \
    -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
WORKDIR /workspace
COPY finetune_dureader.py finetune_poetry.py ./
COPY nlp_gpt3_text-generation_1.3B ./nlp_gpt3_text-generation_1.3B
