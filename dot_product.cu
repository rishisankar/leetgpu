#include "solve.h"
#include <cuda_runtime.h>

// basically the same as reduction.cu

#define FULL_MASK 0xffffffff

constexpr int THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;

constexpr int NUM_PER_THREAD = 96;

__device__ float store[1024];

__global__ void reduce(const float* A, const float* B, int N) {
    __shared__ float sum_per_warp[32];
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int idx = blockIdx.x * THREADS_PER_BLOCK * NUM_PER_THREAD + tid;
    int next_idx = (blockIdx.x+1) * THREADS_PER_BLOCK * NUM_PER_THREAD;
    int bound = min(next_idx, N);
    float f = 0;
    for (int i = idx; i < bound; i += THREADS_PER_BLOCK) {
        f += A[i]*B[i];
    }
    
    for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
        f += __shfl_xor_sync(FULL_MASK, f, i);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {  // tid divisible by warp size
        sum_per_warp[warp_id] = f;
    }
    __syncthreads();

    if (warp_id == 0) {
        f = sum_per_warp[tid];
        for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
    }
    if (tid == 0) {
        store[blockIdx.x] = f;
    }
}

__global__ void reduce_final(float* result, unsigned int nb) {
    __shared__ float sum_per_warp[32];
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    float f = tid < nb ? store[tid] : 0;

    for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
        f += __shfl_xor_sync(FULL_MASK, f, i);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {  // tid divisible by warp size
        sum_per_warp[warp_id] = f;
    }
    __syncthreads();

    if (warp_id == 0) {
        f = sum_per_warp[tid];
        for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
    }
    if (tid == 0) {
        *result = f;
    }
}

void solve(const float* A, const float* B, float* result, int N) {
    const unsigned int nb = (N + THREADS_PER_BLOCK * NUM_PER_THREAD - 1) / THREADS_PER_BLOCK / NUM_PER_THREAD;
    reduce<<<nb, THREADS_PER_BLOCK>>>(A, B, N);
    reduce_final<<<1, 1024>>>(result, nb);
}
