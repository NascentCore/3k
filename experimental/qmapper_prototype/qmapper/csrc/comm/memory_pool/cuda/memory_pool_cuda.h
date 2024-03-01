#pragma once

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include "comm_iface.h"
#include "execute_context/execute_engine.h"
#include "memory_pool/memory_pool.h"
#include "components/cuda_iface/cuda_iface.h"
#include "qmap_locks.h"
#include <cuda_runtime.h>
#include <cuda.h>

#ifndef QMAP_COMM_MEMORY_CONTEXT_CUDA_H
#define QMAP_COMM_MEMORY_CONTEXT_CUDA_H

namespace qmap {
namespace comm {
namespace memory_pool {

typedef struct memory_context_cpu_config {
    size_t elem_size;
    int    max_elems;
} memory_context_cpu_config_t;

std::map<std::string, uint64_t> get_default_cpu_mc_params ();

class CommMemoryPoolContextCuda : public CommMemoryPoolContext {
public:
    CommMemoryPoolContextCuda() {this->name = "memory-pool-context-cpu"; this->prefix = "mpool-cpu";}
    comm_status_t init(std::map<std::string, uint64_t> &params);
    comm_status_t finialize();
    comm_status_t memory_query(const void* ptr, mem_attr_t *attr);
    comm_status_t mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
    comm_status_t mem_pool_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
    comm_status_t mem_free(buffer_header_t *buffer_header);
    comm_status_t memset(void *dst, int value, size_t len);
    comm_status_t flush();

    comm_status_t cuda_mem_alloc_from_pool(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);

public:
    std::string name;
    std::string prefix;
    CommMemoryPool *mpool;
    cudaStream_t stream;
    bool stream_inited;
};

extern CommMemoryPoolContextCuda qmap_mc_cuda;

#define MC_CUDA_INIT_STREAM() do {\
    if(!qmap_mc_cuda.stream_inited) {\
        cudaError_t cuda_st = cudaSuccess;\
        std::lock_guard<utils::SpinLock>(qmap_mc_cuda.lock);\
        if(!qmap_mc_cuda.stream_inited) {\
            cuda_st = cudaStreamCreateWithFlags(&qmap_mc_cuda.stream, cudaStreamNonBlocking);\
            qmap_mc_cuda.stream_inited = true;\
        }\
        if(cuda_st != cudaSuccess) {\
            return cuda_error_to_qmap_status(cuda_st);\
        }\
    }\
} while(0)


}; // namespace memory_pool
}; // namespace comm
}; // namespace qmap

#endif