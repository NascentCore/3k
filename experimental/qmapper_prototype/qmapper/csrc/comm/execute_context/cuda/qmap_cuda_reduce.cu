#include "execute_context/execute_engine.h"
#include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cuda_runtime.h>

#define align_pow2(_n, _p) ((_n) & ((_p) -1))

__device__ inline void add_folat4(float4 &d, const float4 &x, const float4 &y) {
    d.x = x.x + y.x;
    d.y = x.y + y.y;
    d.z = x.z + y.z;
    d.w = x.w + y.w;
}


__global__ void 
reduce_multi_dst_cuda(qmap::comm::execute_engine::task_reduce_multi_dst_t arg) {
    int blocks_per_buffer = gridDim.x / arg.n_bufs;
    int buf_id = blockIdx.x / blocks_per_buffer;
    size_t step = blockDim.x * blocks_per_buffer;
    int idx = threadIdx.x + (blockIdx.x % blocks_per_buffer) * blockDim.x;

    int align = align_pow2((intptr_t)arg.src1[buf_id], 16) |
                align_pow2((intptr_t)arg.src2[buf_id], 16) |
                align_pow2((intptr_t)arg.dst[buf_id], 16);
    if (align == 0) {
        size_t count = arg.counts[buf_id] / 4;
        
    }
}