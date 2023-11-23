nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64  test_cudnn.cu -o test_cudnn -lcudnn
./test_cudnn
