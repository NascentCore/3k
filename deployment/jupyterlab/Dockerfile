FROM registry.cn-beijing.aliyuncs.com/sxwl-ai/torch-base:latest

# 安装 JupyterLab
RUN pip install jupyterlab jupyterlab-language-pack-zh-CN numpy matplotlib graphviz
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/translation-extension && \
    echo '{"locale": "zh_CN"}' > /root/.jupyter/lab/user-settings/@jupyterlab/translation-extension/plugin.jupyterlab-settings

# 设置环境变量以支持中文
ENV LANG zh_CN.UTF-8
ENV JUPYTERLAB_LANGUAGE zh-CN

# 设置工作目录
WORKDIR /workspace

# 启动 JupyterLab
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=$JUPYTER_TOKEN"]
