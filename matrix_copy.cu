#include "solve.h"
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * N; i += blockDim.x * gridDim.x) {
        B[i] = A[i];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
}
