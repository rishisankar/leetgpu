#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int m = threadIdx.y + blockIdx.y * blockDim.y;
    if (m < M && k < K) {
        float ans = 0;
        for (int i = 0; i < N; ++i) {
            ans += A[m * N + i] * B[i * K + k];
        }
        C[m * K + k] = ans;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
