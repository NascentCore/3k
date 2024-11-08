FROM registry.cn-beijing.aliyuncs.com/sxwl-ai/torch-base:v1
WORKDIR /workspace
# Put the layer that will not change frequently on it
RUN python3 -m pip install --no-cache-dir modelscope megatron_util jieba \
    -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html \
    && pip3 install --no-cache-dir -U datasets
COPY chinese-poetry-collection ./chinese-poetry-collection
COPY nlp_gpt3_text-generation_1.3B ./nlp_gpt3_text-generation_1.3B
COPY finetune_dureader.py finetune_poetry.py ./
# https://stackoverflow.com/questions/77433096/notimplementederror-loading-a-dataset-cached-in-a-localfilesystem-is-not-suppor
