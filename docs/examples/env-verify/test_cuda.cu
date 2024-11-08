#include <stdio.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main() {
    int c;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    add<<<1,1>>>(2, 7, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    printf("2 + 7 = %d\n", c);
    return 0;
}
