#include "comm_iface.h"
#include "memory_pool/cuda/memory_pool_cuda.h"
#include "memory_pool/memory_pool.h"
#include "qmap_locks.h"
#include <mutex>

namespace qmap {
namespace comm {

std::map<std::string, uint64_t>
get_default_cuda_mc_params ()
{
    std::map<std::string, uint64_t> default_params;
    default_params["MPOOL_ELEM_SIZE"] = 1024*1024/8;
    default_params["MAX_ELEMS"] = 8;
    return default_params;
}

comm_status_t memory_pool::CommMemoryPoolContextCuda::memset(void *dst, int value, size_t len) {
    cudaError_t st;
    MC_CUDA_INIT_STREAM();
    st = cudaMemsetAsync(dst, value, len, qmap_mc_cuda.stream);
    if(st != cudaSuccess) {
        LOG(ERROR) << "cudaMemset issue failed.";
        return COMM_ERROR;
    }
    st = cudaStreamSynchronize(qmap_mc_cuda.stream);
    if(st != cudaSuccess) {
        LOG(ERROR) << "cudaMemset execute failed.";
        return COMM_ERROR;
    }
    return COMM_OK;
}

comm_status_t memory_pool::CommMemoryPoolContextCuda::memory_query(const void *ptr, mem_attr_t *attr) {
    struct cudaPointerAttributes _attr;
    cudaError_t st;
    CUresult cu_err;
    comm_memory_type_t mem_type;
    void *base_addr;
    size_t alloc_length;

    if(attr->field_mask & MEM_ATTR_FIELD_MEM_TYPE) {
        st = cudaPointerGetAttributes(&_attr, ptr);
        if(st != cudaSuccess) {
            return COMM_ERROR;
        }
        switch(_attr.type) {
            case cudaMemoryTypeHost:
                mem_type = COMM_MEMORY_TYPE_HOST; break;
            case cudaMemoryTypeDevice:
                mem_type = COMM_MEMORY_TYPE_CUDA; break;
            case cudaMemoryTypeManaged:
                mem_type = COMM_MEMORY_TYPE_CUDA_MANAGED; break;
            default:
                mem_type = COMM_MEMORY_TYPE_UNKNOWN;
        }
        attr->mem_type = mem_type;
    }
    if (attr->field_mask & (MEM_ATTR_FIELD_BASE_ADDR|MEM_ATTR_FIELD_ALLOC_LENGTH)) {
        cu_err = cuMemGetAddressRange(reinterpret_cast<CUdeviceptr*>(&base_addr), &alloc_length, reinterpret_cast<CUdeviceptr>(ptr));
        if(cu_err != CUDA_SUCCESS) {
            return COMM_ERROR;
        }

        attr->alloc_length = alloc_length;
        attr->base_addr = base_addr;
    }
    return COMM_OK;
}

comm_status_t cuda_mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mt) {
    cudaError_t st;
    buffer_header_t *header = (buffer_header_t *)malloc(sizeof(buffer_header_t));
    st = (mt == COMM_MEMORY_TYPE_CUDA) ? cudaMalloc(&header->addr, size) : cudaMallocManaged(&header->addr, size);
    header->from_pool = false;
    header->mem_type = mt;
    *header_ptr = header;
    return COMM_OK;
}

comm_status_t 
memory_pool::CommMemoryPoolContextCuda::cuda_mem_alloc_from_pool(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mt) {
    buffer_header_t *h = nullptr;
    if(size <= this->params["MPOOL_ELEM_SIZE"] && mt != COMM_MEMORY_TYPE_CUDA_MANAGED) {
        h = (buffer_header_t *)mpool->mpool_get();
    }
    if (!h) {
        return cuda_mem_alloc(header_ptr, size, mt);
    }
    if(!h->addr) {
        return COMM_ERROR;
    }
    return COMM_OK;
}

comm_status_t 
memory_pool::CommMemoryPoolContextCuda::mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    std::lock_guard<utils::SpinLock> lg(this->lock);
    if(this->params["MAX_ELEMS"] == 0) {
        return cuda_mem_alloc(header_ptr, size, mem_type);
    }
    
}




};
};