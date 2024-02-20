#include "comm_iface.h"
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include <components/cuda_iface/cuda_iface.h>
#include <gtest/gtest.h>
#include "kernels/vector_add.h"

TEST(CudaIfaceTest, StatusTransformation) {
    EXPECT_EQ(COMM_OK, cuda_error_to_qmap_status(cudaSuccess));
    EXPECT_EQ(COMM_ERROR, cuda_error_to_qmap_status(cudaErrorMemoryAllocation));
    float *a = (float *)malloc(3);
    float *b = (float *)malloc(3);
    float *c = (float *)malloc(3);
    EXPECT_EQ(COMM_OK, COMM_CUDA_FUNC(matAdd(a,b,c, 3)));
    free(a);
    free(b);
    free(c);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}