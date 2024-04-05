cmake_minimum_required(VERSION 3.25)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(qmapperc LANGUAGES CXX CUDA)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
add_subdirectory(${THIRD_PARTY_DIR}/pybind11)
add_subdirectory(${THIRD_PARTY_DIR}/json)
add_subdirectory(${THIRD_PARTY_DIR}/glog)
add_subdirectory(${THIRD_PARTY_DIR}/gflags)
add_subdirectory(${THIRD_PARTY_DIR}/googletest)

set(ucx_DIR /usr/local/ucx-1.16)
find_package(ucx REQUIRED)

option(USE_CUDA "Compile CUDA code" ON)
option(USE_ROCM "Compile ROCm code" OFF)
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    add_definitions(-DUSE_CUDA)
endif()
if(USE_ROCM)
    find_package(HIP REQUIRED)
    set(CMAKE_HIP_STANDARD 17)
    add_definitions(-DUSE_ROCM)
endif()

# qmapperc_utils
set(QMAPPER_UTILS
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/qmap_locks.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/qmap_compiler_defs.cc
)
add_library(qmapperc_utils STATIC ${QMAPPER_UTILS})
target_link_libraries(qmapperc_utils PRIVATE glog::glog)
target_include_directories(qmapperc_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# Qmapper Comm
set(QMAPPER_COMM_SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/execute_context/execute_engine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/memory_pool/memory_pool.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/comm/ucx_iface/ucx_iface.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/comm/ucx_iface/ucx_impls.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/comm_iface.cc
)
set(QMAPPER_COMM_CPU_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/components/cpu_iface/cpu_iface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/execute_context/cpu/execute_engine_cpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/memory_pool/cpu/memory_pool_cpu.cc
)
set(QMAPPER_COMM_CUDA_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/components/cuda_iface/cuda_iface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/execute_context/cuda/execute_engine_cuda.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/memory_pool/cuda/memory_pool_cuda.cc
)
set(QMAPPER_COMM_ROCM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/components/rocm_iface/rocm_iface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/execute_context/rocm/execute_engine_rocm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm/memory_pool/rocm/memory_pool_rocm.cc
)

add_library(qmapperc_comm STATIC ${QMAPPER_COMM_SRCS} ${QMAPPER_COMM_CPU_SRCS})
target_include_directories(qmapperc_comm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/comm ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_link_libraries(qmapperc_comm PRIVATE qmapperc_utils glog::glog ucx::ucp ucx::ucs)

if(USE_CUDA)
    target_sources(qmapperc_comm PRIVATE ${QMAPPER_COMM_CUDA_SRCS})
    target_include_directories(qmapperc_comm PRIVATE)
    # set_property(SOURCE ${QMAPPER_COMM_CUDA_SRCS} PROPERTY LANGUAGE CUDA)
    target_link_libraries(qmapperc_comm PRIVATE CUDA::cudart)
endif()

# Unit Tests

if(USE_CUDA)
    ## Comm Cuda Iface Test
    set(CUDA_IFACE_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/comm/components/kernels/vector_add.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/comm/components/cuda_iface_tests.cc
    )
    add_executable(cuda_iface_tests ${CUDA_IFACE_TEST_SRCS})
    # 链接gtest库
    target_include_directories(cuda_iface_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/comm ${CMAKE_CURRENT_SOURCE_DIR}/tests/comm/components)
    target_link_libraries(cuda_iface_tests PRIVATE gtest glog::glog qmapperc_comm ucx::ucp ucx::ucs CUDA::cudart)
endif()


## memory_pool Test
message(STATUS ${CUDA_LIBRARIES})
set(MEMORY_POOL_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/comm/memory_pool/memory_pool_tests.cc
)
if(USE_CUDA)
    set_property(SOURCE ${MEMORY_POOL_TEST_SRCS} PROPERTY LANGUAGE CUDA)
    add_executable(memory_pool_tests ${MEMORY_POOL_TEST_SRCS})
    target_include_directories(memory_pool_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/comm ${CMAKE_CURRENT_SOURCE_DIR}/utils)
    target_link_libraries(memory_pool_tests PRIVATE CUDA::cudart ucx::ucp ucx::ucs glog::glog gtest qmapperc_comm)
    set_target_properties(memory_pool_tests PROPERTIES LINK_FLAGS "-Wl,--unresolved-symbols=ignore-in-object-files")
endif()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA                 : ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA include dirs    : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  CUDA library dir     : ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "")