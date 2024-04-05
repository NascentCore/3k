#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include "comm_iface.h"
#include "execute_context/execute_engine.h"
#include "memory_pool/memory_pool.h"
#include "memory_pool/cpu/memory_pool_cpu.h"

#if USE_CUDA
#include "memory_pool/cuda/memory_pool_cuda.h"
#endif
#if USE_ROCM
#include "memory_pool/cuda/memory_pool_cuda.h"
#endif

TEST(MemoryPoolTests, CorrectInit) {
    CHECK_NE(&qmap_mc_cpu, nullptr);
    std::map<std::string, uint64_t> default_params;
    comm_status_t st = qmap_mc_cpu.init(default_params);
    CHECK_EQ(st, COMM_OK);
    buffer_header_t *header_ptr;
    st = qmap_mc_cpu.mem_alloc(&header_ptr, 1024, COMM_MEMORY_TYPE_HOST);
    CHECK_EQ(st, COMM_OK);
    // CHECK_EQ(header_ptr->from_pool, true);
    // CHECK_NE(header_ptr->addr, nullptr);
    // CHECK_EQ(header_ptr->mem_type, COMM_MEMORY_TYPE_HOST);
    // st = qmap_mc_cpu.mem_free(header_ptr);
    // CHECK_EQ(st, COMM_OK);
#if USE_CUDA
    std::map<std::string, uint64_t> default_cuda_params;
    st = qmap_mc_cuda.init(default_cuda_params);
    // CHECK_EQ(st, COMM_OK);
    // st = qmap_mc_cuda.mem_alloc(&header_ptr, 1024, COMM_MEMORY_TYPE_CUDA);
    // CHECK_EQ(st, COMM_OK);
    // CHECK_EQ(header_ptr->from_pool, true);
    // CHECK_NE(header_ptr->addr, nullptr);
    // CHECK_EQ(header_ptr->mem_type, COMM_MEMORY_TYPE_CUDA);
    // mem_attr_t attr{};
    // attr.field_mask &= MEM_ATTR_FIELD_MEM_TYPE;
    // attr.field_mask &= (MEM_ATTR_FIELD_BASE_ADDR|MEM_ATTR_FIELD_ALLOC_LENGTH);
    // st = qmap_mc_cuda.memory_query(header_ptr->addr, &attr);
    // CHECK_EQ(attr.alloc_length, 1024);
    // CHECK_EQ(attr.base_addr, header_ptr->addr);
    // CHECK_EQ(attr.mem_type, COMM_MEMORY_TYPE_CUDA);
    // st = qmap_mc_cuda.mem_free(header_ptr);
    // CHECK_EQ(st, COMM_OK);
#endif
#if USE_ROCM
#endif
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}