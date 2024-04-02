#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int num_devs, driver_version;
    cudaGetDeviceCount(&num_devs);
    std::cout << "This machine has " << num_devs << " devices." << std::endl;

    cuDriverGetVersion(&driver_version);
    std::cout << "The version of cuda of this machine is " << driver_version << std::endl;
    CUdevice cu_dev;
    cuCtxGetDevice(&cu_dev);
    int attr;
    cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, cu_dev);
    if (attr & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) {
        std::cout << "This machine support GDR write." << std::endl;
    }
    else {
        std::cout << "This machine does not support GDR write." << std::endl;
    }
}