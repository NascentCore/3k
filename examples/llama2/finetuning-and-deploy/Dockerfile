FROM swr.cn-east-3.myhuaweicloud.com/sxwl/torch-base:latest
WORKDIR /workspace
RUN pip3 install --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple packaging
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple  torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118
#RUN MAX_JOBS=4 pip install flash-attn==2.3.0 --no-build-isolation
COPY Fastchat ./
RUN  pip3 install "fschat[model_worker,train]"