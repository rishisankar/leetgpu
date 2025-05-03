#include "solve.h"
#include <cuda_runtime.h>

/*
TODOs:

- Similar to the sorting.cu file, this is not optimized at all yet.

- There are probably more interesting algorithms if K is guaranteed
  to be much smaller than N. But here, K can be as big as N and
  N can be up to 1e8, and we need to output in descending order 
  so this is effectively a superset of the sorting problem...

- Top K seems to have a better approach with bitonic sort,
  this odd-even merge sort could have a similar idea too.
  https://anilshanbhag.com/static/papers/gputopk_sigmod18.pdf


  A general idea: once we have sorted chunks of size >= k,
  when we do the merge step we can discard the bottom half.
  We can also merge sorted chunks of size k fairly easily within
  a block: similar to merge sorted lists cpu algorithm, have each
  thread maintain a chunk of size k, use warp reduction to get
  the max and advance that thread's pointer.
*/

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
            if (f1 < f2) {
                arr[idx1] = f2;
                arr[idx2] = f1;
            }
        }
    }
}

__global__ void copy_k_kernel(const float* input, float* output, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < k) {
        output[tid] = input[tid];
    }
}

// data is device pointer
void solve(const float* input, float* output, int N, int k) {

    float* inputCpy; // since input is const ptr (not ideal..)
    cudaMalloc((void**)&inputCpy, N * sizeof(float));
    copy_k_kernel<<<(N+1023)/1024, 1024>>>(input, inputCpy, N);

    int Npower2 = 1;
    while (Npower2 < N) {
        Npower2 *= 2;
    }
    // first we sort in descending order
    for (int A = 1; A < Npower2; A *= 2) {
        for (int S = A; S >= 1; S /= 2) {
            int num_blocks = Npower2 / (2 * A);
            int comparisons_per_block = A == S ? A : A - S;
            int total_comps = num_blocks * comparisons_per_block;
            constexpr int threads_per_block = 1024;
            odd_even_merge_step_kernel<<<(total_comps + threads_per_block -  1) / threads_per_block, threads_per_block>>>(N, A, S, inputCpy, num_blocks, comparisons_per_block);
        }
    }
    copy_k_kernel<<<(k+1023)/1024, 1024>>>(inputCpy, output, k);
    cudaFree(inputCpy);

}
