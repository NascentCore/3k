#pragma once

#include <cuda.h>
#include <ratio>


#ifndef QMAP_EXCUTE_ENGINE_CUDA
#define QMAP_EXCUTE_ENGINE_CUDA

namespace qmap {
namespace comm {
namespace execute_engine {

typedef enum ec_cuda_stream_task_mode {
    EC_CUDA_TASK_KERNEL,
    EC_CUDA_TASK_MEM_OPS,
    EC_CUDA_TASK_AUTO,
    EC_CUDA_TASK_LAST,
} ec_cuda_stream_task_mode_t;

typedef enum ec_cuda_executor_state {
    EC_CUDA_EXECUTOR_INITIALIZED,
    EC_CUDA_EXECUTOR_POSTED,
    EC_CUDA_EXECUTOR_STARTED,
    EC_CUDA_EXECUTOR_SHUTDOWN,
    EC_CUDA_EXECUTOR_SHUTDOWN_ACK,
} ec_cuda_executor_state_t;


}; // namespace execute_engine
}; // namespace comm
}; // namespace qmap

#endif