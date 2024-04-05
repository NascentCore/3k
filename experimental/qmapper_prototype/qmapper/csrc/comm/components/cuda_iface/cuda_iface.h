#pragma once

#include "comm_iface.h"
#include <glog/logging.h>

#if USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

static inline comm_status_t cuda_error_to_qmap_status(cudaError_t cuda_status) {
    comm_status_t status;

    switch(cuda_status) {
    case cudaSuccess:
        status = COMM_OK;
        break;
    case cudaErrorNotReady:
        status = COMM_INPROGRESS;
        break;
    default:
        status = COMM_ERROR;
    }
    return status;
}

#define COMM_CUDA_FUNC(_func) \
    ({\
        comm_status_t qmap_status;\
        do {\
            cudaError_t qmap_result = (_func);\
            if(cudaSuccess != qmap_result) {\
                LOG(ERROR) << #_func << " failed " << cudaGetErrorString(qmap_result);\
            }\
            qmap_status = cuda_error_to_qmap_status(qmap_result);\
        } while(0);\
        qmap_status;\
    })

#define COMM_CUDADRV_FUNC(_func) \
   ({\
        comm_status_t qmap_status = COMM_OK;\
        do {\
            CUresult qmap_result = (_func);\
            const char *cu_err_str;\
            if(CUDA_SUCCESS != qmap_result) {\
                cuGetErrorString(qmap_result, &cu_err_str);\
                LOG(ERROR) << #_func << " failed " << cu_err_str;\
                qmap_status = COMM_ERROR;\
            }\
        } while(0);\
        qmap_status;\
    })

#endif