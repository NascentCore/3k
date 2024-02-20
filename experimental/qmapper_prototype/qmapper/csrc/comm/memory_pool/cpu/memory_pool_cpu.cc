#include "memory_pool/cpu/memory_pool_cpu.h"
#include "memory_pool/memory_pool.h"
#include "comm_iface.h"
#include <cstdint>
#include <cstdlib>
#include <map>
#include <glog/logging.h>
#include <mutex>
#include <ucs/datastruct/mpool.h>

namespace qmap{
namespace comm {
namespace memory_pool {

std::map<std::string, uint64_t>
get_default_cpu_mc_params ()
{
    std::map<std::string, uint64_t> default_params;
    default_params["MPOOL_ELEM_SIZE"] = 1024*1024/8;
    default_params["MAX_ELEMS"] = 8;
    return default_params;
}







comm_status_t CommMemoryPoolContextCpu::init(std::map<std::string, uint64_t> &params) {
    return COMM_OK;
}

comm_status_t CommMemoryPoolContextCpu::mem_alloc(buffer_header_t **header_ptr, size_t size, comm_memory_type_t mem_type) {
    size_t size_with_header = size + sizeof(buffer_header_t);
    buffer_header_t *h = (buffer_header_t *)malloc(size_with_header);
    if (!h) {
        LOG(ERROR) << "malloc failed";
        return COMM_ERROR;
    }
    h->from_pool = 0;
    h->addr = PTR_OFFSET(h, sizeof(buffer_header_t));
    h->mem_type = mem_type;
    LOG(INFO) << "allocated " << size << " bytes host memory";
    return COMM_OK;
}

}; // namespace memory_pool
}; // namespace comm
}; // namespace qmap