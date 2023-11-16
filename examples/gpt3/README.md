# Install ModelScope library core framework

\# https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install modelscope

# Env vars

export NCCL\_DEBUG=WARN NCCL\_P2P\_LEVEL=SYS NCCL\_IB\_CUDA\_SUPPORT=1 NCCL\_DEBUG\_SUBSYS=ALL NCCL\_IB\_GDR\_LEVEL=SYS NCCL\_NET\_GDR\_LEVEL=SYS NCCL\_NET\_GDR\_READ=1 NCCL\_NET=IB

# Single node

\#torchrun --standalone --nproc\_per\_node 8 --nnodes=1 --node\_rank=0 --master\_addr= --master\_port= modelscope\_sxwl\_gpt3.py
torchrun --standalone --nproc\_per\_node 8 --nnodes=1 modelscope\_sxwl\_gpt3.py
