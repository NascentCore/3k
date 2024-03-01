#pragma once

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include "comm_iface.h"
#include "qmap_locks.h"
#include "execute_context/execute_engine.h"
#include <ucs/datastruct/mpool.h>

#ifndef QMAP_COMM_MEMORY_CONTEXT_H
#define QMAP_COMM_MEMORY_CONTEXT_H

typedef struct mem_attr {
    uint64_t field_mask;
    comm_memory_type_t mem_type;
    void *base_addr;
    size_t alloc_length;
} mem_attr_t;

typedef struct buffer_header {
    comm_memory_type_t mem_type;
    void *addr;
    bool from_pool;
} buffer_header_t;

typedef enum mem_attr_field {
    MEM_ATTR_FIELD_MEM_TYPE = GET_BIT(0),
    MEM_ATTR_FIELD_BASE_ADDR = GET_BIT(1),
    MEM_ATTR_FIELD_ALLOC_LENGTH = GET_BIT(2),
} mem_attr_field_t;

namespace qmap {
namespace comm {
namespace memory_pool {

class CommMemoryPool;
class CommMemoryPoolContext;

static ucs_status_t mpool_chunk_alloc_wrapper(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);
static void mpool_chunk_release_wrapper(ucs_mpool_t *mp, void *chunk);
static void mpool_obj_init_wrapper(ucs_mpool_t *mp, void *obj, void *chunk);
static void mpool_obj_cleanup_wrapper(ucs_mpool_t *mp, void *obj);

class CommMemoryPool {
public:
    CommMemoryPool();
    ~CommMemoryPool();
    comm_status_t init(size_t priv_size, size_t elem_size, 
                  size_t align_offset, size_t alignment, 
                  unsigned int elems_per_chunk, unsigned int max_elems,
                  thread_mode_t thread_mode, std::string &name
                  );
    void cleanup();
    comm_status_t hugetlb_malloc(size_t *size_p, void **chunk_p);
    void hugetlb_free(void *chunk);
    void *mpool_get();
    void mpool_put(void *obj);
    virtual comm_status_t chunk_alloc(size_t *size_p, void **chunk_p);
    virtual void chunk_release(void *chunk);
    virtual void obj_init(void *obj, void *chunk);
    virtual void obj_cleanup(void *obj);
public:
    std::string name;
    ucs_mpool_params_t ucs_mpool_params;
    ucs_mpool_t ucs_mpool;
    thread_mode_t thread_mode;
    utils::SpinLock lock;
};


class CommMemoryPoolContext {
public:
    static comm_status_t check_available(comm_memory_type_t mem_type);
    comm_status_t global_init(std::map<std::string, uint64_t> &params);
    comm_status_t global_finialize();
    comm_status_t available(comm_memory_type_t mem_type);

    virtual comm_status_t init(std::map<std::string, uint64_t> &params);
    virtual comm_status_t finialize();
    virtual comm_status_t memory_query(const void* ptr, mem_attr_t *attr);
    virtual comm_status_t mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type);
    virtual comm_status_t mem_free(buffer_header_t *buffer_header);
    virtual comm_status_t memset(void *dst, int value, size_t len);
    virtual comm_status_t memcpy(void *dst, int value, size_t len);
    virtual comm_status_t flush();
    

public:
    static std::vector<CommMemoryPoolContext *> memory_contexts;
    std::string name;
    uint32_t ref_cnt;
    utils::SpinLock lock;
    ee_type_t ee_type;
    comm_memory_type_t memory_type;
    std::map<std::string, uint64_t> params;
};

}; // namespace memory_pool
}; // namespace comm
}; // namespace qmap

#endif