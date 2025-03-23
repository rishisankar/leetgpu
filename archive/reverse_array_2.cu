#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    __shared__ float input_store[1024];
    int already_reversed_width = 512 * blockIdx.x;
    int num_to_load = min(N/2 - already_reversed_width, 512);
    int ti = threadIdx.x;
    if (ti < num_to_load) {
        input_store[ti] = input[ti + already_reversed_width];
    } else if (ti >= 1024 - num_to_load) {
        input_store[ti] = input[N - already_reversed_width - 1024 + ti];
    }
    __syncthreads();
    if (ti < num_to_load) {
        input[ti + already_reversed_width] = input_store[1023 - ti];
    } else if (ti >= 1024 - num_to_load) {
        input[N - already_reversed_width - 1024 + ti] = input_store[1023 - ti];
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
