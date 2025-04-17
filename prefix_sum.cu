#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

__device__ float store[1024*32];

__device__ float s1[1024], s2[1024];

template<bool store_value>
__device__ void prefix_sum_compute(const float* input, float* output, int N, float* s) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int block_id = blockIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    s[tid] = 0;
    __syncthreads();
    
    int loop_bound = (N + 31);
    loop_bound -= (loop_bound % 32);
    for (int i = tid; i < loop_bound; i += num_threads) {
        float f = i < N ? input[i] : 0;
        // sum over warp
        for (int i = 16; i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
        // store the sum of these 32 values
        if (lane_id == 0) {
            s[i/32] = f;
        }
    }
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

    for (int i = tid; i < loop_bound; i += num_threads) {
        float f = i < N ? input[i] : 0;
        for (int d = 1; d <= 16; d *= 2) {
           float _f = __shfl_up_sync(FULL_MASK, f, d);
           if (lane_id - d >= 0) f += _f;
        }
        if (i < N) {
            if (i >= 32) {
                f += s[i/32 - 1];
            }
            output[i] = f;
        }
    }
    // for (int i = tid * 32; i < min(N, (tid+1)*32); i++) {
    //     float ans = input[i];
    //     if (i % 32 != 0) {
    //         ans += output[i-1];
    //     }
    //     if (tid > 0) {
    //         ans += s[i/32 - 1];
    //     }
    //     output[i] = ans;
    // }

    if constexpr (store_value) {
        if (tid == 0) {
            store[block_id] = output[N-1];
        }
    }
}

// template<bool store_value>
// __device__ void prefix_sum_compute(const float* input, float* output, int N, float* s) {
//     int tid = threadIdx.x;
//     int block_id = blockIdx.x;
//     int start = tid * 32;
//     if (start < N) {
//         output[start] = input[start];
//         for (int i = start + 1; i < min(N, start + 32); i++) {
//             output[i] = output[i-1] + input[i];
//         }
//     }
//     __syncthreads();
//     if (tid == 0) {
//         for (int i = 32+31; i < N; i += 32) {
//             output[i] += output[i-32];
//         }
//     }
//     __syncthreads();

//     for (int i = start; i < min(N, start + 31); i++) {
//         if (tid != 0) {
//             output[i] += output[start - 1];
//         }
//     }

//     if constexpr (store_value) {
//         store[block_id] = output[N - 1];
//     }
// }

// prefix sum small chunks of the overall array of size NUM_THREADS * 32.
__global__ void prefix_sum_kernel1(const float* input, float* output, int N) {
    // extern __shared__ float s[]; // shared memory, size intended to be N block / 32

    int num_per_block = blockDim.x * 32;
    int block_id = blockIdx.x;
    int N_this_block = min(num_per_block, N - num_per_block * block_id);
    prefix_sum_compute<true>(input + num_per_block * block_id, output + num_per_block * block_id, N_this_block, s1);

}

// prefix sum over store
__global__ void prefix_sum_kernel2(int N_store) {
    extern __shared__ float s[]; // shared memory, size intended to be N / 32
    prefix_sum_compute<false>(store, store, N_store, s);
}


// add store's sums to each element
__global__ void prefix_sum_kernel3(float* output, int N) {
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int num_threads = blockDim.x;
    int num_per_block = num_threads * 32;
    int loop_end = min(N, num_per_block * (block_id + 1));
    // first block is already done
    if (block_id > 0) {
        int store_val = store[block_id - 1];
        for (int i = num_per_block * block_id + tid; i < loop_end; i += num_threads) {
            output[i] += store_val;
        }
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int num_threads = 1024;
    int num_blocks = (N + (32*num_threads - 1)) / (32*num_threads);
    prefix_sum_kernel1<<<num_blocks, 1024>>>(input, output, N);
    prefix_sum_kernel2<<<1, 1024, num_threads * sizeof(float)>>>(num_blocks);
    prefix_sum_kernel3<<<num_blocks, 1024>>>(output, N);
} 
