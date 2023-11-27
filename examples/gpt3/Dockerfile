FROM registry.ap-southeast-1.aliyuncs.com/sxwl-ai/cuda_base:2023-10-23
WORKDIR /workspace
# Put the layer that will not change frequently on it
RUN python3 -m pip install  --no-cache-dir -U datasets==2.15.0 modelscope megatron_util jieba \
    -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html \
    && pip3 install -U datasets
COPY chinese-poetry-collection ./chinese-poetry-collection
COPY nlp_gpt3_text-generation_1.3B ./nlp_gpt3_text-generation_1.3B
COPY finetune_dureader.py finetune_poetry.py ./
# https://stackoverflow.com/questions/77433096/notimplementederror-loading-a-dataset-cached-in-a-localfilesystem-is-not-suppor
