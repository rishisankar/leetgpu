#include "solve.h"
#include <cuda_runtime.h>

// TODO: right now this makes O(log^2 n) kernel calls, where each does many global memory swaps in parallel.
// This hasn't been optimized to take advantage of coalescing, etc

__global__ void odd_even_merge_step_kernel(int N, int A, int S, float* arr, int num_blocks, int comparisons_per_block) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = tid / comparisons_per_block;
    int comparison_id = tid % comparisons_per_block;
    int block_start = block_id * 2 * A;
    if (tid < num_blocks * comparisons_per_block) {
        int idx1, idx2;
        if (A == S) {
            idx1 = block_start + comparison_id;
            idx2 = idx1 + A;
        } else {
            int start = block_start + S;
            int comparison_chunk_width_id = comparison_id / S;
            start += 2 * S * comparison_chunk_width_id;
            int comparison_id_in_chunk = comparison_id % S;
            idx1 = start + comparison_id_in_chunk;
            idx2 = idx1 + S;
        }
        if (idx2 < N) {
            float f1 = arr[idx1], f2 = arr[idx2];
            if (f1 > f2) {
                arr[idx1] = f2;
                arr[idx2] = f1;
            }
        }
    }
}

// data is device pointer
void solve(float* data, int N) {
    int Npower2 = 1;
    while (Npower2 < N) {
        Npower2 *= 2;
    }
    for (int A = 1; A < Npower2; A *= 2) {
        for (int S = A; S >= 1; S /= 2) {
            int num_blocks = Npower2 / (2 * A);
            int comparisons_per_block = A == S ? A : A - S;
            int total_comps = num_blocks * comparisons_per_block;
            constexpr int threads_per_block = 1024;
            odd_even_merge_step_kernel<<<(total_comps + threads_per_block -  1) / threads_per_block, threads_per_block>>>(N, A, S, data, num_blocks, comparisons_per_block);
        }
    }

}
