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

#ifndef QMAP_COMM_MEMORY_POOL_CPU_H
#define QMAP_COMM_MEMORY_POOL_CPU_H

typedef struct comm_memory_context_cpu_config {
    size_t elem_size;
    int    max_elems;
} comm_memory_context_cpu_config_t;

namespace qmap {
namespace comm {
namespace memory_pool {

comm_status_t cpu_mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
comm_status_t cpu_mem_alloc_from_pool(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);

class CommMemoryPoolCpu : public CommMemoryPool {
public:
    CommMemoryPoolCpu() = default;
    ~CommMemoryPoolCpu() = default;
    comm_status_t chunk_alloc(size_t *size_p, void **chunk_p);
    void chunk_release(void *chunk);
    void obj_init(void *obj, void *chunk);
    void obj_cleanup(void *obj);
public:
    std::string prefix;

};


// params field
// MPOOL_ELEM_SIZE: size of each elemen in mc_cpu
// MPOOL_MAX_ELEMS: max amount of elements in mc cpu
// THREAD_MODE: thread mode of the context
class CommMemoryPoolContextCpu : public CommMemoryPoolContext {
public:
    CommMemoryPoolContextCpu();
    ~CommMemoryPoolContextCpu();
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
    CommMemoryPoolCpu mpool;
    utils::SpinLock lock;
    ee_type_t ee_type;
    comm_memory_type_t memory_type;
    std::map<std::string, uint64_t> params;
    thread_mode_t thread_mode;

    size_t elem_size;
    size_t max_elems;

    bool inited;
};

}; // namespace memory_pool
}; // namespace comm
}; // namespace qmap

extern qmap::comm::memory_pool::CommMemoryPoolContextCpu qmap_mc_cpu;

#endif