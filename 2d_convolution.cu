#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

constexpr int THREADS_PER_WARP = 32;

__global__ void conv_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int output_rows,
    int output_cols
) {
    __shared__ float kernel_shared[32*32];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int threads_per_block = blockDim.x;
    int warp_id = tid / THREADS_PER_WARP;
    int lane_id = tid % THREADS_PER_WARP;
    int warps_per_block = threads_per_block / THREADS_PER_WARP;
    int output_id = bid * warps_per_block + warp_id;

    for (int i = tid; i < kernel_rows * kernel_cols; i += threads_per_block) {
        kernel_shared[i] = kernel[i];
    }
    __syncthreads();

    for (int i = output_id; i < output_rows * output_cols; i += gridDim.x * warps_per_block) {
        int row = i / output_cols;
        int col = i % output_cols;
        float sum = 0;
        for (int j = lane_id; j < kernel_rows * kernel_cols; j += THREADS_PER_WARP) {
            int kernel_row = j / kernel_cols;
            int kernel_col = j % kernel_cols;
            // todo: too many global memory accesses to input...
            sum += kernel_shared[j] * input[(row + kernel_row) * input_cols + col + kernel_col];
        }
        for (int j = THREADS_PER_WARP / 2; j >= 1; j >>= 1) {
            sum += __shfl_xor_sync(FULL_MASK, sum, j);
        }
        output[i] = sum;
    }
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    const int threads_per_block = 1024;
    const int warps_per_block = threads_per_block / THREADS_PER_WARP;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    const int num_blocks = (output_rows * output_cols + warps_per_block - 1) / warps_per_block;
    conv_kernel<<<num_blocks, threads_per_block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, output_rows, output_cols);
}