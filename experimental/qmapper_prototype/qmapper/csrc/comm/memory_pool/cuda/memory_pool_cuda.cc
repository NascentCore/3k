#include <mutex>

#include "comm_iface.h"
#include "qmap_locks.h"
#include "memory_pool/memory_pool.h"
#include "memory_pool/cuda/memory_pool_cuda.h"
#include "components/cuda_iface/cuda_iface.h"
#include "components/cpu_iface/cpu_iface.h"

// cuda memory pool
comm_status_t
qmap::comm::memory_pool::CommMemoryPoolCuda::chunk_alloc(size_t *size_p, void **chunk_p) {
    *chunk_p = malloc(*size_p);
    if(!*chunk_p) {
        LOG(ERROR) << "Failed to allocated " << *size_p << " bytes.";
        return COMM_ERROR;
    }
    return COMM_OK;
}

void
qmap::comm::memory_pool::CommMemoryPoolCuda::chunk_release(void *chunk) {
    free(chunk);
}

void
qmap::comm::memory_pool::CommMemoryPoolCuda::obj_init(void *obj, void *chunk) {
    buffer_header_t *h = (buffer_header_t *)obj;
    h->from_pool = true;
    h->addr = nullptr;
    comm_status_t st = COMM_CUDA_FUNC(cudaMalloc(&h->addr, this->elem_size));
    if(st != COMM_OK) {
        cudaGetLastError();
    }
    h->mem_type = COMM_MEMORY_TYPE_CUDA;
}

void
qmap::comm::memory_pool::CommMemoryPoolCuda::obj_cleanup(void *obj) {
    buffer_header_t *h = (buffer_header_t *)obj;
    if(h->addr) {
        comm_status_t st = COMM_CUDA_FUNC(cudaFree(h->addr));
        if(st != COMM_OK) {
            cudaGetLastError();
        }
    }
    h->addr = nullptr;
}

// cuda memory pool context

qmap::comm::memory_pool::CommMemoryPoolContextCuda::CommMemoryPoolContextCuda() 
{
    this->name = "QMAPPER_COMM_MEMORY_POOL_CUDA";
    this->prefix = "QMAP_COMM_MPOLL_CUDA";
    this->ref_cnt = 0;
}

qmap::comm::memory_pool::CommMemoryPoolContextCuda::~CommMemoryPoolContextCuda() {
    this->flush();
    this->finialize();
}

qmap::comm::memory_pool::CommMemoryPoolContextCuda qmap_mc_cuda{};

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCuda::init(std::map<std::string, uint64_t> &params) {
    std::lock_guard<utils::SpinLock> lg(lock);
    for (auto it : params) {
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

    int num_devices, driver_version;
    comm_status_t status;
    status = COMM_CUDA_FUNC(cudaGetDeviceCount(&num_devices));
    if (status != COMM_OK) {
        LOG(ERROR) << "Cannot get cuda device count.";
        return COMM_ERROR;
    }
    status = COMM_CUDADRV_FUNC(cuDriverGetVersion(&driver_version));
    if (status != COMM_OK) {
        LOG(ERROR) << "Cannot get cuda driver version.";
        return COMM_ERROR;
    }
    if(driver_version >= 11030) {
        CUdevice cu_dev;
        int attr = 0;
        status = COMM_CUDADRV_FUNC(cuCtxGetDevice(&cu_dev));
        if (status != COMM_OK) {
            LOG(ERROR) << "Failed to get the cuda device.";
            return COMM_ERROR;
        }
        COMM_CUDADRV_FUNC(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, cu_dev));
        if (attr & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) {
            COMM_CUDADRV_FUNC(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, cu_dev));
            if (CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER > attr) {
                flush_op = cuda_mem_flush_to_owner;
            } else {
                flush_op = cuda_mem_flush_no_op;
            }
        }
    } else {
        DLOG(ERROR) << "Does not support old cuda version.";
        return COMM_ERROR;
    }
    status = COMM_CUDADRV_FUNC(cuCtxGetCurrent(&cu_ctx));
    status = mpool.init(0, sizeof(buffer_header_t), 
                        0, CACHE_LINE_SIZE, 1, 
                        this->max_elems, 
                        this->thread_mode,
                        "MPOOL_CUDA");
    mpool.elem_size = this->elem_size;

    if (status != COMM_OK) {
        LOG(ERROR) << "Failed to create memory pool for cuda.";
        return COMM_ERROR;
    }
    status = COMM_CUDA_FUNC(cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking));
    if (status != COMM_OK) {
        LOG(ERROR) << "Failed to create cuda stream.";
        this->mpool.cleanup();
        return COMM_ERROR;
    }
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCuda::finialize() {
    CUcontext tmp_ctx;
    cuCtxPushCurrent(this->cu_ctx);
    mpool.cleanup();
    COMM_CUDA_FUNC(cudaStreamDestroy(stream));
    cuCtxPopCurrent(&tmp_ctx);
    return COMM_OK;
}

comm_status_t 
qmap::comm::memory_pool::CommMemoryPoolContextCuda::memset(void *dst, int value, size_t len) {
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
    DLOG(INFO) << "cudaMemset succeed.";
    return COMM_OK;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCuda::memcpy(void *dst, void *src, size_t len, comm_memory_type_t dst_mem_type, comm_memory_type_t src_mem_type) {
    if (!inited) {
        return COMM_INITIALIZING;
    }
    comm_status_t status;
    if( (src_mem_type != COMM_MEMORY_TYPE_CUDA && 
         src_mem_type != COMM_MEMORY_TYPE_CUDA_MANAGED ) ||
         src_mem_type != dst_mem_type ) {
        return COMM_ERROR;
    }
    status = COMM_CUDA_FUNC(cudaMemcpyAsync(dst, src, len, cudaMemcpyDefault, stream));
    if (status != COMM_OK) {
        LOG(ERROR) << "fail to launch cuda memcpy.";
        return COMM_ERROR;
    }
    return COMM_CUDA_FUNC(cudaStreamSynchronize(stream));
}

comm_status_t 
qmap::comm::memory_pool::CommMemoryPoolContextCuda::memory_query(const void *ptr, mem_attr_t *attr) {
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

comm_status_t 
qmap::comm::memory_pool::cuda_mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    buffer_header_t *header = (buffer_header_t *)malloc(sizeof(buffer_header_t));
    (mem_type == COMM_MEMORY_TYPE_CUDA) ? cudaMalloc(&header->addr, size) : cudaMallocManaged(&header->addr, size);
    header->from_pool = false;
    header->mem_type = mem_type;
    *header_ptr = header;
    DLOG(INFO) << "Allocated cuda memory " << size << " bytes with cuda malloc.";
    return COMM_OK;
}

comm_status_t 
qmap::comm::memory_pool::cuda_mem_alloc_from_pool(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mt) {
    size_t size_with_header = sizeof(buffer_header_t) + size;
    buffer_header_t *header = nullptr;
    if(size_with_header <= qmap_mc_cuda.max_elems && mt != COMM_MEMORY_TYPE_CUDA_MANAGED) {
        header = (buffer_header_t *)qmap_mc_cuda.mpool.mpool_get();
    }
    if (!header) {
        return qmap::comm::memory_pool::cuda_mem_alloc(header_ptr, size, mt);
    }
    header->from_pool = true;
    header->addr = PTR_OFFSET(header, sizeof(buffer_header_t));
    header->mem_type = COMM_MEMORY_TYPE_CUDA;
    *header_ptr = header;
    DLOG(INFO) << "allocated " << size_with_header << " bytes from cuda pool.";
    return COMM_OK;
}

comm_status_t 
qmap::comm::memory_pool::CommMemoryPoolContextCuda::mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    std::lock_guard<utils::SpinLock> lg(this->lock);
    if(this->params["MAX_ELEMS"] == 0) {
        return qmap::comm::memory_pool::cuda_mem_alloc(header_ptr, size, mem_type);
    }
    return qmap::comm::memory_pool::cuda_mem_alloc_from_pool(header_ptr, size, mem_type);
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCuda::mem_free(buffer_header_t *buffer_header) {
    if (buffer_header->from_pool) {
        qmap_mc_cuda.mpool.mpool_put(buffer_header);
        return COMM_OK;
    } else {
        cudaError_t status = cudaFree(buffer_header);
        if(status != cudaSuccess) {
            LOG(ERROR) << "Failed to dealloc cuda memory, error: " << cudaGetErrorString(status);
            return COMM_ERROR;
        }
        return COMM_OK;
    }
}

comm_status_t
qmap::comm::memory_pool::cuda_mem_flush_no_op() {
    DLOG(ERROR) << "Cuda device does not support flush operation.";
    return COMM_ERROR;
}

comm_status_t
qmap::comm::memory_pool::cuda_mem_flush_to_owner() {
    DLOG(ERROR) << "Cuda device does not support flush operation.";
    return COMM_ERROR;
}

comm_status_t
qmap::comm::memory_pool::CommMemoryPoolContextCuda::flush() {
    if(inited) {
        return flush_op();
    } else {
        return cuda_mem_flush_no_op();
    }
}


