#include "solve.h"
#include <cuda_runtime.h>

__global__ void batched_matmul_kernel(const float* A_gbl, const float* B_gbl, float* C_gbl, int M, int N, int K) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    // each block does one batch
    const float* A = A_gbl + block_idx * M * K;
    const float* B = B_gbl + block_idx * K * N;
    float* C = C_gbl + block_idx * M * N;

    // naive matmul, todo optimize
    for (int i = tid; i < M * N; i += blockDim.x) {
        int r = i / N, c = i % N;
        float ans = 0;
        for (int j = 0; j < K; j++) {
            ans += A[r * K + j] * B[j * N + c];
        }
        C[i] = ans;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    const int threadsPerBlock = 1024;
    batched_matmul_kernel<<<BATCH, threadsPerBlock>>>(A, B, C, M, N, K);
}
