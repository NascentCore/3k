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

#ifndef QMAP_COMM_MEMORY_CONTEXT_CPU_H
#define QMAP_COMM_MEMORY_CONTEXT_CPU_H

namespace qmap {
namespace comm {
namespace memory_pool {

typedef struct memory_context_cpu_config {
    size_t elem_size;
    int    max_elems;
} memory_context_cpu_config_t;

std::map<std::string, uint64_t> get_default_cpu_mc_params ();

class CommMemoryPoolCpu : public CommMemoryPool {
public:
    void *allocate(size_t size);
    void deallocate(void *addr, size_t size);
    comm_status_t chunk_alloc(size_t *size_p, void **chunk_p);
    void chunk_release(void *chunk);
    void obj_init(void *obj, void *chunk);
    void obj_cleanup(void *obj);
public:
    std::string prefix;

};

class CommMemoryPoolContextCpu : public CommMemoryPoolContext {
public:
    CommMemoryPoolContextCpu() {this->name = "memory-pool-context-cpu"; this->prefix = "mpool-cpu";}
    comm_status_t init(std::map<std::string, uint64_t> &params);
    comm_status_t finialize();
    comm_status_t memory_query(const void* ptr, mem_attr_t *attr);
    comm_status_t mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
    comm_status_t mem_pool_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
    comm_status_t mem_free(buffer_header_t *buffer_header);
    comm_status_t memset(void *dst, int value, size_t len);
    comm_status_t flush();

public:
    std::string name;
    std::string prefix;
    CommMemoryPool *mpool;
};

}; // namespace memory_pool
}; // namespace comm
}; // namespace qmap

#endif