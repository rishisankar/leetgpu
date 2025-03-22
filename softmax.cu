#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

constexpr int THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int REDUCTION_SIZE = THREADS_PER_BLOCK / WARP_SIZE;

// using ideas from https://www.youtube.com/watch?v=IpHjDoW4ffw

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float max_per_warp[REDUCTION_SIZE];
    __shared__ float sum_per_warp[REDUCTION_SIZE];
    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int warp_id = tid / WARP_SIZE;

    // Step 1: compute max(input)

    // Step 1a: calculate max per thread
    float thread_max = -INFINITY;
    for (int i = tid; i < N; i += n_threads) {
        thread_max = fmaxf(thread_max, input[i]);
    }
    // Step 1b: Max across warp (register reduction)
    for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(FULL_MASK, thread_max, i));
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {  // tid divisible by warp size
        max_per_warp[warp_id] = thread_max;
    }
    // Step 1c: sync - can't proceed without finishing each warp
    __syncthreads();
    // Step 1d: Max across max_per_warp
    if (warp_id == 0) {
        // works since REDUCTION_SIZE == WARP_SIZE
        thread_max = max_per_warp[tid];
        for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(FULL_MASK, thread_max, i));
        }
    }
    // Step 1e: Store overall max so all threads can access
    if (tid == 0) {
        max_per_warp[0] = thread_max;
    }
    // Step 1f: sync again - so all threads can pickup the actual max
    __syncthreads();
    thread_max = max_per_warp[0];

    // Step 2: compute softmax denominator

    // Step 2a: calculate sum per thread
    float thread_sum = 0;
    for (int i = tid; i < N; i += n_threads) {
        thread_sum += __expf(input[i] - thread_max);
    }
    // Step 2b: sum across warp (register reduction)
    for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
        thread_sum += __shfl_xor_sync(FULL_MASK, thread_sum, i);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {  // tid divisible by warp size
        sum_per_warp[warp_id] = thread_sum;
    }
    // Step 2c: sync - can't proceed without finishing each warp
    __syncthreads();
    // Step 2d: Sum across sum_per_warp
    if (warp_id == 0) {
        // works since REDUCTION_SIZE == WARP_SIZE
        thread_sum = sum_per_warp[tid];
        for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
            thread_sum += __shfl_xor_sync(FULL_MASK, thread_sum, i);
        }
    }
    // Step 2e: Store overall sum so all threads can access
    if (tid == 0) {
        sum_per_warp[0] = thread_sum;
    }
    // Step 2f: sync again - so all threads can pickup the sum
    __syncthreads();
    thread_sum = sum_per_warp[0];

    // Step 3: compute outputs
    for (int i = tid; i < N; i += n_threads) {
        output[i] = __expf(input[i] - thread_max) / thread_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    softmax_kernel<<<1, THREADS_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}

