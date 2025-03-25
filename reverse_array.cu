#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N/2) {
        float tmp = input[idx];
        input[idx] = input[N-1-idx];
        input[N-1-idx] = tmp;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N/2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
}
