#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    float ans = 0;
    if (threadIdx.x < kernel_size) {
        ans = input[blockIdx.x + threadIdx.x] * kernel[threadIdx.x];
    }
    __shared__ float reduction[32];
    int max_reduction = (kernel_size + 31) / 32;
    for (int i = 16; i >= 1; i >>= 1) {
        ans += __shfl_xor_sync(0xffffffff, ans, i);
    }
    if (threadIdx.x % 32 == 0) {
        reduction[threadIdx.x / 32] = ans;
    }
    __syncthreads();
    if (threadIdx.x < max_reduction) {
        ans = reduction[threadIdx.x];
        for (int i = 16; i >= 1; i >>= 1) {
            ans += __shfl_xor_sync(0xffffffff, ans, i);
        }
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = ans;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = kernel_size;
    int blocksPerGrid = output_size;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
