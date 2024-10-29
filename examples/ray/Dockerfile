FROM sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/ray:2.37.0.1b620f-py310-cu121
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.6.1.post2
COPY vllm_app.py /app/vllm_app.py
COPY __init__.py /app/__init__.py