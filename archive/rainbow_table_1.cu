#include "solve.h"
#include <cuda_runtime.h>

constexpr int NUM_BLOCKS = 112;

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R, int numsPerBlock) {
    int bd = min((blockIdx.x+1)*numsPerBlock, N);
    for (int start = blockIdx.x * numsPerBlock + threadIdx.x; start < bd; start += 1024) {
        unsigned int hash = fnv1a_hash(input[start]);
        for (int i = 1; i < R; i++) {
            hash = fnv1a_hash(hash);
        }
        output[start] = hash;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 1024;
    int numsPerBlock = (N + NUM_BLOCKS - 1) / NUM_BLOCKS;
    fnv1a_hash_kernel<<<NUM_BLOCKS, threadsPerBlock>>>(input, output, N, R, numsPerBlock);
}
