#include <cuda_runtime.h>

__global__ void add_one_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1.0f;
}

void launch_add_one(float* d_data, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_one_kernel<<<grid, block>>>(d_data, n);
}
