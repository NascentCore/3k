FROM sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple accelerate peft bitsandbytes transformers trl tensorboard

COPY train.py ./

CMD ["python", "train.py"]