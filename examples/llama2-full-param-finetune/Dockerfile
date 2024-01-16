FROM sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/torch-base:v2024-01-12-01

RUN pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes

WORKDIR /workspace
COPY *.py ./
