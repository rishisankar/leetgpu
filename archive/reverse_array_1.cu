#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int ti = threadIdx.x;
    int idx;
    int warp_id = ti / 32;
    int lane_id = ti & 31;
    bool bad = false;
    if (lane_id >= 16) {
        // access from the back
        idx = N - 512 * blockIdx.x - 16 * (warp_id + 2) + lane_id;
        if (idx < (N+1)/2) {
            bad = true;
        }
    } else {
        idx = 512 * blockIdx.x + warp_id * 16 + lane_id;
        if (idx >= N/2) {
            bad = true;
        }
    }
    float array_val = bad ? 0 : input[idx];
    array_val = __shfl_sync(0xffffffff, array_val, 31-lane_id);
    if (!bad) {
        input[idx] = array_val;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
