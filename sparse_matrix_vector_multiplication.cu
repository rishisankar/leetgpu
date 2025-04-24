#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
constexpr int WARP_SIZE = 32;

__global__ void kernel(const float* A, const float* x, float* y, int M, int N) {
    __shared__ float sum_per_warp[32];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) sum_per_warp[warp_id] = 0;

    // loop end should be the first multiple of 32 >= N
    int loopEnd = N + 31;
    loopEnd -= loopEnd%32;

    for (int i = tid; i < loopEnd; i += blockDim.x) {
        float f = i < N ? A[blockIdx.x * N + i] : 0;
        if (f != 0) {
            f *= x[i];
        }
        // warp reduce to find sum over warp
        for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
        if (lane_id == 0) {
            sum_per_warp[warp_id] += f;
        }
    }
    __syncthreads();
    if (warp_id == 0) {
        float f = sum_per_warp[lane_id];
        for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
        if (lane_id == 0) {
            y[blockIdx.x] = f;
        }
    }
}

// A, x, y are device pointers
void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int num_threads = 1024;
    int num_blocks = M;
    // TODO figure out how to use nnz?
    kernel<<<num_blocks, num_threads>>>(A, x, y, M, N);
} 