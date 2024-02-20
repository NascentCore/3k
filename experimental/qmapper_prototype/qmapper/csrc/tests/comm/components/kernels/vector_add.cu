#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels/vector_add.h"

template<typename T>
__global__ void matAdd_cuda(T *a,T *b,T *sum)
{
    int i = blockIdx.x*blockDim.x+ threadIdx.x;
    sum[i] = a[i] + b[i];
}


cudaError_t matAdd(float *a,float *b,float *sum, int length)
{
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);
    int threadMaxSize = devProp.maxThreadsPerBlock;
    int blockSize = (length+threadMaxSize-1)/threadMaxSize;
    dim3 block(blockSize);
    int size = length * sizeof(float);
    float *sumGPU,*aGPU,*bGPU;
    cudaMalloc((void**)&sumGPU,size);
    cudaMalloc((void**)&aGPU,size);
    cudaMalloc((void**)&bGPU,size);
    cudaMemcpy((void*)aGPU,(void*)a,size,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bGPU,(void*)b,size,cudaMemcpyHostToDevice);
    matAdd_cuda<float><<<block,size/blockSize>>>(aGPU,bGPU,sumGPU);
    cudaMemcpy(sum,sumGPU,size,cudaMemcpyDeviceToHost);
    cudaFree(sumGPU);
    cudaFree(aGPU);
    return cudaFree(bGPU);
}