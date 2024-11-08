# 介绍
检测机器是否安装完成了model scope运行所需的环境

## cuda
nvcc test_cuda.cu -o test_cuda
./test_cuda

## cuDNN
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64  test_cudnn.cu -o test_cudnn -lcudnn
./test_cudnn

## model scope 运行
进入相应的python环境然后开始检测.

### model scope 推理
python hello_modelscope_inference.py

### model scope 单机单卡
python hello_modelscope_train_single.py

### model scope 单机多卡
 torchrun --nproc_per_node=4 --master_port=9527 ./hello_modelscope_train_multi.py

