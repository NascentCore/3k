#include <cuda.h>
#include <cuda_runtime.h>

extern "C" cudaError_t matAdd(float *a,float *b, float *c, int length);