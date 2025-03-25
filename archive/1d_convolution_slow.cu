#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

// amount of input elements to load into shared memory per block
// needs to be at least kernel size
constexpr int INPUT_LOAD_PER_BLOCK = 2048;

constexpr int WARP_SIZE = 32;

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size, int outputs_per_block, int output_size) {
    int bi = blockIdx.x;
    int start = bi * outputs_per_block;
    outputs_per_block = min(outputs_per_block, output_size - blockIdx.x * outputs_per_block);
    // load kernel into shared memory upfront to avoid repeated global memory access
    __shared__ float shared_kernel[1024];
    int ti = threadIdx.x;
    if (ti < kernel_size) {
        shared_kernel[ti] = kernel[ti];
    }
    int warp_id = ti / WARP_SIZE;

    // load elements from input into shared memory too
    __shared__ float shared_input[INPUT_LOAD_PER_BLOCK];    
    for (int i = ti; i < min(INPUT_LOAD_PER_BLOCK, input_size - start); i += blockDim.x) {
        shared_input[i] = input[start + i];
    }
    __syncthreads();

    __shared__ float reduced_sum[32];
    for (int i = 0; i < outputs_per_block; i += blockDim.x) {
        float stored_ans_per_thread = 0;
        int last = min(blockDim.x, outputs_per_block - i);
        for (int j = 0; j < last; ++j) {
            float ans_per_thread = 0;
            if (ti < kernel_size) {
                ans_per_thread = shared_input[i + j + ti] * shared_kernel[ti];
            }
            for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
                ans_per_thread += __shfl_xor_sync(FULL_MASK, ans_per_thread, i);
            }
            if ((ti & (WARP_SIZE - 1)) == 0) {  // ti divisible by warp size
                reduced_sum[warp_id] = ans_per_thread;
            }
            __syncthreads();
            if (warp_id == j/32) {
                ans_per_thread = reduced_sum[ti&31];
                for (int i = (WARP_SIZE>>1); i >= 1; i >>= 1) {
                    ans_per_thread += __shfl_xor_sync(FULL_MASK, ans_per_thread, i);
                }
            }
            __syncthreads();
            if (j == ti) {
                stored_ans_per_thread = ans_per_thread;
            }
        }
        if (ti < last) {
            output[start + i + ti] = stored_ans_per_thread;
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 1024;
    int outputs_per_block = INPUT_LOAD_PER_BLOCK - kernel_size + 1;
    int blocksPerGrid = (output_size + outputs_per_block - 1) / outputs_per_block;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size, outputs_per_block, output_size);
}
