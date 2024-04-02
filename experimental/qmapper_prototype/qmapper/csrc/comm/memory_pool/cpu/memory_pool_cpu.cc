#include <cstdlib>
#include <cstring>
#include <glog/logging.h>
#include <mutex>

#include "comm_iface.h"
#include "execute_context/execute_engine.h"
#include "qmap_locks.h"
#include "memory_pool/memory_pool.h"
#include "memory_pool/cpu/memory_pool_cpu.h"
#include "components/cpu_iface/cpu_iface.h"

// cpu mpool
comm_status_t 
qmap::comm::memory_pool::CommMemoryPoolCpu::chunk_alloc(size_t *size_p, void **chunk_p) {
    *chunk_p = malloc(*size_p);
    if(!*chunk_p) {
        LOG(ERROR) << "Failed to allocated " << *size_p << " bytes.";
        return COMM_ERROR;
    }
    return COMM_OK;
}

void
qmap::comm::memory_pool::CommMemoryPoolCpu::chunk_release(void *chunk) {
    free(chunk);
}

void
qmap::comm::memory_pool::CommMemoryPoolCpu::obj_init(void *obj, void *chunk) {
    buffer_header_t *h = (buffer_header_t *)obj;
    h->from_pool = true;
    h->addr = PTR_OFFSET(obj, sizeof(buffer_header_t));
    h->mem_type = COMM_MEMORY_TYPE_HOST; 
}

void
qmap::comm::memory_pool::CommMemoryPoolCpu::obj_cleanup(void *obj) {
}




// memory context

qmap::comm::memory_pool::CommMemoryPoolContextCpu::CommMemoryPoolContextCpu() 
{
    this->name = "QMAPPER_COMM_MEMORY_POOL_CPU";
    this->prefix = "QMAP_COMM_MPOLL_CPU";
    this->ref_cnt = 0;
}

qmap::comm::memory_pool::CommMemoryPoolContextCpu::~CommMemoryPoolContextCpu() {
    this->flush();
    this->finialize();
}

qmap::comm::memory_pool::CommMemoryPoolContextCpu qmap_mc_cpu{};

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::init(std::map<std::string, uint64_t> &params) {
    std::lock_guard<utils::SpinLock> lg(lock);
    for(auto it : params) {
        this->params.insert(it);
    }
    if (params.count("MAX_ELEMS") > 0) {
        this->max_elems = params["MAX_ELEMS"];
    } else {
        this->max_elems = 1024*1024;
    }
    if (params.count("MPOOL_ELEM_SIZE") > 0) {
        this->max_elems = params["MPOOL_ELEM_SIZE"];
    } else {
        this->max_elems = 8;
    }
    if (params.count("THREAD_MODE") > 0) {
        this->max_elems = params["THREAD_MODE"];
    } else {
        this->max_elems = THREAD_FUNNELED;
    }
    this->memory_type = COMM_MEMORY_TYPE_HOST;
    mpool.init(0, sizeof(buffer_header_t) + this->elem_size,
               0, 
               CACHE_LINE_SIZE, 
               1, 
               this->max_elems, 
               this->thread_mode, 
               "QMAPPER_MPOOL_CPU");
    this->ee_type = EE_CPU_THREAD;
    this->inited = true;
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::finialize() {
    std::lock_guard<utils::SpinLock> lg(lock);
    if(inited) {
        this->mpool.cleanup();
        inited = false;
    }
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::memory_query(const void *ptr, mem_attr_t *attr) {
    DLOG(ERROR) << "Is it necessary to query a main memory buffer?";
    return COMM_ERROR;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::memset(void *dst, int value, size_t len) {
    memset(dst, value, len);
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::memcpy(void *dst, void *src, size_t len, comm_memory_type_t dst_mem_type, comm_memory_type_t src_mem_type) {
    if(dst_mem_type != src_mem_type || src_mem_type != COMM_MEMORY_TYPE_HOST) {
        LOG(ERROR) << "wrong memory type";
        return COMM_ERROR;
    }
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::cpu_mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    size_t size_with_header = sizeof(buffer_header_t) + size;
    buffer_header_t *header = (buffer_header_t *)malloc(size_with_header);
    if(!header) {
        LOG(ERROR) << "cpu alloc " << size_with_header << " bytes failed.";
        return COMM_ERROR;
    }
    header->from_pool = false;
    header->addr = PTR_OFFSET(header, size_with_header);
    header->mem_type = COMM_MEMORY_TYPE_HOST;
    *header_ptr = header;
    DLOG(INFO) << "allocated " << size_with_header << " bytes with malloc.";
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::cpu_mem_alloc_from_pool(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    size_t size_with_header = sizeof(buffer_header_t) + size;
    buffer_header_t *header = nullptr;
    if(size_with_header <= qmap_mc_cpu.elem_size) {
        header = (buffer_header_t*)qmap_mc_cpu.mpool.mpool_get();
    }
    if (!header) {
        return qmap::comm::memory_pool::cpu_mem_alloc(header_ptr, size, mem_type);
    }
    header->from_pool = true;
    header->addr = PTR_OFFSET(header, sizeof(buffer_header_t));
    header->mem_type = COMM_MEMORY_TYPE_HOST;
    *header_ptr = header;
    DLOG(INFO) << "allocated " << size_with_header << " bytes from host pool.";
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    std::lock_guard<utils::SpinLock> lg(this->lock);
    if(this->max_elems == 0) {
        return qmap::comm::memory_pool::cpu_mem_alloc(header_ptr, size, mem_type);
    }
    return qmap::comm::memory_pool::cpu_mem_alloc_from_pool(header_ptr, size, mem_type); 
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::mem_free(buffer_header_t *buffer_header) {
    if(buffer_header->from_pool) {
        mpool.mpool_put(buffer_header);
    } else {
        free(buffer_header);
    }
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCpu::flush() {
    DLOG(ERROR) << "No need to flush.";
    return COMM_ERROR;
}