#pragma once

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include "qmap_locks.h"
#include "comm_iface.h"
#include "execute_context/execute_engine.h"
#include "memory_pool/memory_pool.h"
#include "components/cuda_iface/cuda_iface.h"
#include <cuda_runtime.h>
#include <cuda.h>

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


namespace qmap {
namespace comm {
namespace memory_pool {

comm_status_t cuda_mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
comm_status_t cuda_mem_alloc_from_pool(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
comm_status_t cuda_mem_flush_no_op();
comm_status_t cuda_mem_flush_to_owner();

class CommMemoryPoolCuda : public CommMemoryPool {
public:
    CommMemoryPoolCuda() = default;
    ~CommMemoryPoolCuda() = default;
    comm_status_t chunk_alloc(size_t *size_p, void **chunk_p);
    void chunk_release(void *chunk);
    void obj_init(void *obj, void *chunk);
    void obj_cleanup(void *obj);
public:
    std::string prefix;
    size_t elem_size;
};

class CommMemoryPoolContextCuda : public CommMemoryPoolContext {
public:
    CommMemoryPoolContextCuda();
    ~CommMemoryPoolContextCuda();
    comm_status_t init(std::map<std::string, uint64_t> &params);
    comm_status_t finialize();
    comm_status_t memory_query(const void* ptr, mem_attr_t *attr);
    comm_status_t mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
    comm_status_t mem_free(buffer_header_t *buffer_header);
    comm_status_t memset(void *dst, int value, size_t len);
    comm_status_t memcpy(void *dst, void *src, size_t len, comm_memory_type_t dst_mem_type, comm_memory_type_t src_mem_type);
    comm_status_t flush();
public:
    std::string prefix;
    CommMemoryPoolCuda mpool;
    CUcontext cu_ctx;
    cudaStream_t stream;
    bool stream_inited;
    utils::SpinLock lock;
    ee_type_t ee_type;
    comm_memory_type_t memory_type;
    std::map<std::string, uint64_t> params;
    thread_mode_t thread_mode;

    size_t elem_size;
    size_t max_elems;
    std::function<comm_status_t()> flush_op; 
    bool inited;
};


}; // namespace memory_pool
}; // namespace comm
}; // namespace qmap

extern qmap::comm::memory_pool::CommMemoryPoolContextCuda qmap_mc_cuda;
