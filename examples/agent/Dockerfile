# 使用官方Python运行环境作为基础镜像
FROM --platform=linux/amd64  python:3.8-slim

# 设置工作目录为/app
WORKDIR /app

# 将当前目录下的所有Python文件复制到容器中的/app目录
COPY ./*.py /app/

# 安装Flask及其他可能需要的库
RUN pip install Flask  requests

# 配置环境变量
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 暴露端口5000供外部访问
EXPOSE 5000

# 容器启动时执行的命令，运行Flask服务
CMD ["flask", "run"]
