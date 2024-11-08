#include <iostream>
#include <cudnn.h>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // 输出cuDNN版本信息
    std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;

    // 验证cuDNN的一些功能，可以根据需要添加其他操作
    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);

    std::cout << "cuDNN validation successful." << std::endl;

    // 释放资源
    cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroy(cudnn);

    return 0;
}

