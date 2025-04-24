#include "solve.h"
#include <cuda_runtime.h>

constexpr int NUM_BLOCKS = 8;

__global__ void set_zero_kernel(int* histogram, int num_bins) {
    int tid = threadIdx.x;
    if (tid < num_bins) {
        histogram[tid] = 0;
    }
}

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    __shared__ float cts[1024];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_per_block = (N + NUM_BLOCKS - 1)  / NUM_BLOCKS;
    cts[tid] = 0;
    __syncthreads();
    for (int i = num_per_block * bid + tid; i < min(N, num_per_block * (bid+1)); i += 1024) {
        atomicAdd(cts + input[i], 1);
    }
    __syncthreads();
    if (tid < num_bins) {
        atomicAdd(histogram + tid, cts[tid]);
    }
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    set_zero_kernel<<<1,1024>>>(histogram, num_bins);
    histogram_kernel<<<NUM_BLOCKS, 1024>>>(input, histogram, N, num_bins);
}
