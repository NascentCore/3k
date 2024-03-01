#include "comm_iface.h"
#include "config.h"
#include "memory_pool/memory_pool.h"
#include "qmap_locks.h"
#include "qmap_compiler_defs.h"
#include <glog/logging.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <ucs/datastruct/mpool.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <vector>

namespace qmap {
namespace comm {

std::array<std::string, 6> memory_type_names = {
    "host",
    "cuda",
    "cuda-managed",
    "rocm",
    "rocm-managed",
    "unknown",
};

std::array<std::string, 3> execute_engine_type_names = {
    "cuda-stream",
    "cpu-thread",
    "rocm-stream",
};

static ucs_status_t
memory_pool::mpool_chunk_alloc_wrapper(ucs_mpool_t *mp, size_t *size_p, void **chunk_p) {
    memory_pool::CommMemoryPool *mpool = class_container_of(memory_pool::CommMemoryPool, ucs_mpool, mp);
    comm_status_t st;
    st = mpool->chunk_alloc(size_p, chunk_p);
    return qmap_status_to_ucs_status(st);
}

static void
memory_pool::mpool_chunk_release_wrapper(ucs_mpool_t *mp, void *chunk) {
    memory_pool::CommMemoryPool *mpool = class_container_of(memory_pool::CommMemoryPool, ucs_mpool, mp);
    mpool->chunk_release(chunk);
}

static void
memory_pool::mpool_obj_init_wrapper(ucs_mpool_t *mp, void *obj, void *chunk) {
    memory_pool::CommMemoryPool *mpool = class_container_of(memory_pool::CommMemoryPool, ucs_mpool, mp);
    mpool->obj_init(obj, chunk);
}

static void
memory_pool::mpool_obj_cleanup_wrapper(ucs_mpool_t *mp, void *obj) {
    memory_pool::CommMemoryPool *mpool = class_container_of(memory_pool::CommMemoryPool, ucs_mpool, mp);
    mpool->obj_cleanup(obj);
}

comm_status_t 
memory_pool::CommMemoryPool::hugetlb_malloc(size_t *size_p, void **chunk_p) {
    ucs_status_t st;
    st = ucs_mpool_hugetlb_malloc(&this->ucs_mpool, size_p, chunk_p);
    return ucs_status_to_qmap_status(st);
}

void 
memory_pool::CommMemoryPool::hugetlb_free(void *chunk) {
    ucs_mpool_hugetlb_free(&this->ucs_mpool, chunk);
}

comm_status_t 
memory_pool::CommMemoryPool::chunk_alloc(size_t *size_p, void **chunk_p) {
    return memory_pool::CommMemoryPool::hugetlb_malloc(size_p, chunk_p);
}

void 
memory_pool::CommMemoryPool::chunk_release(void *chunk) {
    memory_pool::CommMemoryPool::hugetlb_free(chunk);
}

void 
memory_pool::CommMemoryPool::obj_init(void *obj, void *chunk) {
    return;
}

void 
memory_pool::CommMemoryPool::obj_cleanup(void *obj) {
    return;
}

comm_status_t 
memory_pool::CommMemoryPool::init(size_t priv_size, size_t elem_size, 
                                  size_t align_offset, size_t alignment, 
                                  unsigned int elems_per_chunk, 
                                  unsigned int max_elems, 
                                  thread_mode_t thread_mode, std::string &name) {
    std::lock_guard<utils::SpinLock> lock_guard(this->lock);
    ucs_mpool_ops_t *ucs_ops = (ucs_mpool_ops_t *)calloc(1, sizeof(ucs_mpool_ops_t));
    if (! ucs_ops) {
        LOG(ERROR) << "Failed to alloc " << sizeof(ucs_mpool_ops_t) << " bytes";
        return COMM_ERROR;
    }
    this->thread_mode = thread_mode;
    ucs_ops->chunk_alloc = memory_pool::mpool_chunk_alloc_wrapper;
    ucs_ops->chunk_release = memory_pool::mpool_chunk_release_wrapper;
    ucs_ops->obj_init = memory_pool::mpool_obj_init_wrapper;
    ucs_ops->obj_cleanup = memory_pool::mpool_obj_cleanup_wrapper;
    
    ucs_mpool_params_t init_params;
    ucs_mpool_params_reset(&init_params);
    init_params.priv_size = priv_size;
    init_params.align_offset = align_offset;
    init_params.alignment = alignment;
    init_params.elem_size = elem_size;
    init_params.elems_per_chunk = elems_per_chunk;
    init_params.max_elems = max_elems;
    init_params.ops = ucs_ops;
    init_params.name = name.c_str();
    init_params.max_chunk_size = ((size_t) -1);
    init_params.grow_factor = 1.0;
    this->ucs_mpool_params = init_params;
    
    ucs_status_t st = ucs_mpool_init(&this->ucs_mpool_params, &this->ucs_mpool);
    return ucs_status_to_qmap_status(st);
}

void memory_pool::CommMemoryPool::cleanup() {
    void *ops = (void *)this->ucs_mpool.data->ops;
    ucs_mpool_cleanup(&this->ucs_mpool, 0);
    free(ops);
}

void *
memory_pool::CommMemoryPool::mpool_get() {
    std::lock_guard<utils::SpinLock> lg(this->lock);
    return ucs_mpool_get(&this->ucs_mpool);
}

void
memory_pool::CommMemoryPool::mpool_put(void *obj) {
    std::lock_guard<utils::SpinLock> lg(this->lock);
    ucs_mpool_put(obj);
}



// memory pool contexts

comm_status_t memory_pool::CommMemoryPoolContext::check_available(comm_memory_type_t mem_type) {
    if(ucs_unlikely(memory_contexts[mem_type] == nullptr)) {
        LOG(ERROR) << "not supported memory type: " << memory_type_names[mem_type]; 
        return COMM_ERROR;
    }
    return COMM_OK;
}

comm_status_t memory_pool::CommMemoryPoolContext::global_init(std::map<std::string, uint64_t> &params) {
    int n_mcs;
    memory_pool::CommMemoryPoolContext *memory_context;
    comm_status_t status;
    n_mcs = CommConfig::mc_libs.size();
    for (int i = 0; i < n_mcs; i++) {
        memory_context = reinterpret_cast<memory_pool::CommMemoryPoolContext *>(&CommConfig::mc_libs[i]);
        status = memory_context->init(params);
        if (COMM_OK != status) {
            LOG(ERROR) << "failed to init memory context: " << memory_context->name;
            continue;
        }
        memory_context->ref_cnt ++;
        memory_contexts[i] = memory_context;
    }
    return COMM_OK;
}




}; // namespace comm
}; // namespace qmap