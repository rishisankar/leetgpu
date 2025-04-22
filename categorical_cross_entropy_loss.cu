#include "solve.h"
#include <cuda_runtime.h>

__device__ float sample_loss[10000];

#define FULL_MASK 0xffffffff

__global__ void sample_loss_kernel(const float* logits, const int* true_labels, int N, int C) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // loop end should be the first multiple of 32 >= C
    int loopEnd = C + 31;
    loopEnd -= loopEnd%32;

    __shared__ float sum_per_warp[33];

    float f_tot = 0;
    for (int i = tid; i < loopEnd; i += blockDim.x) {
        float f = i < C ? __expf(logits[bid * C + i]) : 0;
        for (int i = 16; i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
        f_tot += f;
    }
    if (lane_id == 0) {
        sum_per_warp[warp_id] = f_tot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float f = sum_per_warp[lane_id];
        for (int i = 16; i >= 1; i >>= 1) {
            f += __shfl_xor_sync(FULL_MASK, f, i);
        }
        if (lane_id == 0) {
            // 1 to avoid bank conflict with 32
            sum_per_warp[1] = logf(f);
        }
    } else if (warp_id == 1 && lane_id == 0) {
        sum_per_warp[32] = logits[bid * C + true_labels[bid]];
    }
    __syncthreads();
    if (tid == 0) {
        sample_loss[bid] = sum_per_warp[1] - sum_per_warp[32];
    }
}

__global__ void sum_kernel(float* out, int N) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float loss = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        loss += sample_loss[i];
    }
    __shared__ float sum_per_warp[32];
    for (int i = 16; i >= 1; i >>= 1) {
        loss += __shfl_xor_sync(FULL_MASK, loss, i);
    }
    if (lane_id == 0) {
        sum_per_warp[warp_id] = loss;
    }
    __syncthreads();
    if (warp_id == 0) {
        loss = sum_per_warp[lane_id];
        for (int i = 16; i >= 1; i >>= 1) {
            loss += __shfl_xor_sync(FULL_MASK, loss, i);
        }
        if (lane_id == 0) {
            *out = loss / N;
        }
    }
}

// logits, true_labels, loss are device pointers
void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    // each block handles one sample
    // then a single block sums the sample losses to get final loss
    sample_loss_kernel<<<N, 1024>>>(logits, true_labels, N, C);
    sum_kernel<<<1, 1024>>>(loss, N);
}