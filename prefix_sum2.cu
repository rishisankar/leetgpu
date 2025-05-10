#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

__device__ float store1[512*512];
__device__ float store2[512];

__device__ void prefix_sum_compute(const float* input, float* output, int N, float* s, float* storer) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int block_id = blockIdx.x;

    s[tid] = tid < N ? input[tid] : 0;
    __syncthreads();

    // up sweep
    int offset = 1;
    for (int d = 512; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int a = (tid+1) * (offset * 2) - 1 - offset;
            int b = (tid+1) * (offset * 2) - 1;
            s[b] += s[a];
        }
        offset *= 2;
    }

    // down sweep
    for (int d = 2; d < 1024; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d - 1) {
            int a = (tid+1) * offset - 1;
            int b = (tid+1) * offset - 1 + offset/2;
            s[b] += s[a];
        }
    }
    __syncthreads();

    if (tid < N) output[tid] = s[tid];

    if (storer && tid == 0) {
        storer[block_id] = s[N-1];
    }
}

__global__ void prefix_sum_kernel1(const float* input, float* output, int N) {
    extern __shared__ float s[];

    int num_per_block = 512;
    int block_id = blockIdx.x;
    int N_this_block = min(num_per_block, N - num_per_block * block_id);
    prefix_sum_compute(input + num_per_block * block_id, output + num_per_block * block_id, N_this_block, s, store1);

}

// prefix sum over store
__global__ void prefix_sum_kernel2(int N_store1) {
    extern __shared__ float s[]; // shared memory, size intended to be N / 32

    int num_per_block = 512;
    int block_id = blockIdx.x;
    int N_this_block = min(num_per_block, N_store1 - num_per_block * block_id);
    prefix_sum_compute(store1 + num_per_block * block_id, store1 + num_per_block * block_id, N_this_block, s, store2);
}


// add store's sums to each element
__global__ void prefix_sum_kernel3(float* output, int N) {
    // TODO
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int num_threads = 512;
    int num_blocks = (N + num_threads - 1) / num_threads;
    prefix_sum_kernel1<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(input, output, N);
    int num_blocks2 = (num_blocks + num_threads - 1) / num_threads;
    prefix_sum_kernel2<<<num_blocks2, num_threads, num_threads * sizeof(float)>>>(num_blocks);
    prefix_sum_kernel3<<<num_blocks, 1024>>>(output, N);
} 
