#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BW = 64;
constexpr int NUM_WARPS = 16;

/*
Approach:
- uses tensor cores to multiply 16x16 matrices
- 16 warps per block each handling a 16x16 region
- blocks handle a 64x64 tile of output
*/

template <bool isKEven>
__global__ void matmul_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half SA[BW * BW];
    __shared__ half SB[BW * BW];
    __shared__ float SC[BW * BW];

    float zero;
    half *ptr_zero = reinterpret_cast<half*>(&zero);
    ptr_zero[0] = __float2half(0.0f);
    ptr_zero[1] = __float2half(0.0f);
    float* SAfloat = reinterpret_cast<float *>(SA);
    float* SBfloat = reinterpret_cast<float *>(SB);
    const float* Afloat = reinterpret_cast<const float *>(A);
    const float* Bfloat = reinterpret_cast<const float *>(B);

    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 4;
    int warpCol = warpId % 4;

    int numBlocksN = (N+BW-1)/BW;
    int numBlocksK = (K+BW-1)/BW;
    
    int blockId = blockIdx.x;
    int blockX = blockId / numBlocksN;
    int blockY = blockId % numBlocksN;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int iter = 0; iter < numBlocksK; iter++) {
        // if k is even, able to load 2 halfs a time (20% speedup on nsight compute)
        if constexpr (isKEven) {
            // load block of A into shared memory
            for (int i = tid; i < BW * BW / 2; i += blockDim.x) {
                int row = i / (BW/2), col = i % (BW/2);
                row += blockX * BW;
                col += iter * (BW/2);
                int Aidx = row * K / 2 + col;
                SAfloat[i] = (Aidx >= M * K / 2) ? zero : Afloat[Aidx];
            }
            // load block of B into shared memory
            for (int i = tid; i < BW * BW / 2; i += blockDim.x) {
                int row = i / (BW/2), col = i % (BW/2);
                row += iter * BW;
                col += blockY * (BW/2);
                int Bidx = row * N / 2 + col;
                SBfloat[i] = (Bidx >= N * K / 2) ? zero : Bfloat[Bidx];
            }
        } else {
            // load block of A into shared memory
            for (int i = tid; i < BW * BW; i += blockDim.x) {
                int row = i / BW, col = i % BW;
                row += blockX * BW;
                col += iter * BW;
                SA[i] = (row >= M || col >= K) ? __float2half(0.0f) : A[row * K + col];
            }
            // load block of B into shared memory
            for (int i = tid; i < BW * BW; i += blockDim.x) {
                int row = i / BW, col = i % BW;
                row += iter * BW;
                col += blockY * BW;
                SB[i] = (row >= K || col >= N) ? __float2half(0.0f) : B[row * N + col];
            }
        }
        // make sure all loads finished
        __syncthreads();
        
        half *SA_warp_ptr = SA + 16 * BW * warpRow;
        half *SB_warp_ptr = SB + 16 * warpCol;
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(a_frag, SA_warp_ptr, BW);
            wmma::load_matrix_sync(b_frag, SB_warp_ptr, BW);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            SA_warp_ptr += 16;
            SB_warp_ptr += 16 * BW;
        }
    }

    float *SC_warp_ptr = SC + 16 * BW * warpRow + 16 * warpCol;
    wmma::store_matrix_sync(SC_warp_ptr, c_frag, BW, wmma::mem_row_major);

    // finally store back to global
    __syncthreads();  // make sure all stores finished
    for (int i = tid; i < BW * BW; i += blockDim.x) {
        int row = i / BW;
        int col = i % BW;
        row += blockX * BW;
        col += blockY * BW;
        if (row < M && col < N) {
            C[row * N + col] = __float2half(beta * __half2float(C[row * N + col]) + alpha * SC[i]);
        }
    }
}

// A, B, and C are device pointers
void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int numBlocksM = (M + BW -1) / BW;
    int numBlocksN = (N + BW - 1) / BW;
    if (K % 2 == 0) {
        matmul_kernel<true><<<numBlocksM * numBlocksN, 32 * NUM_WARPS>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        matmul_kernel<false><<<numBlocksM * numBlocksN, 32 * NUM_WARPS>>>(A, B, C, M, N, K, alpha, beta);
    }
}