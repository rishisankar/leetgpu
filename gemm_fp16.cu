#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int BM = 128, BK = 32, BN = 64;
constexpr int MMA_WIDTH = 16; // assuming MMA multiplies 16x16
constexpr int WM = 2, WN = 2; // width of each warp tile
const int NUM_WARPTILES_X = BM / MMA_WIDTH / WM;
const int NUM_WARPTILES_Y = BN / MMA_WIDTH / WN;
constexpr int NUM_WARPS = NUM_WARPTILES_X * NUM_WARPTILES_Y;

__global__ void matmul_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    extern __shared__ half S[];
    half *SA = S;  // size BM x BK
    half *SB = S + BM * BK;  // size BK x BN
    float *SC = reinterpret_cast<float*>(SB + BK * BN);  // size BM x BN

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int warpX = warpId / NUM_WARPTILES_Y;
    int warpY = warpId % NUM_WARPTILES_Y;

    int numBlocksN = (N + BN - 1) / BN;
    int numBlocksK = (K + BK - 1) / BK;

    // each block computes a single block of output
    int blockId = blockIdx.x;
    int blockX = blockId / numBlocksN;
    int blockY = blockId % numBlocksN;

    // initialize fragments (these will use register space)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WN];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WM * WN];
    for (int i = 0; i < WM * WN; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    for (int iter = 0; iter < numBlocksK; iter++) {
        // load into SA
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int blockRow = i / BK, blockCol = i % BK;
            int globalRow = blockRow + blockX * BM;
            int globalCol = blockCol + iter * BK;
            SA[i] = (globalRow >= M || globalCol >= K) ? __float2half(0.0f) : A[globalRow * K + globalCol];
        }
        // load into SB
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int blockRow = i / BN, blockCol = i % BN;
            int globalRow = blockRow + iter * BK;
            int globalCol = blockCol + blockY * BN;
            SB[i] = (globalRow >= K || globalCol >= N) ? __float2half(0.0f) : B[globalRow * N + globalCol];
        }
        // sync threads to ensure all blocks have loaded their part
        __syncthreads();

        // do matmuls with tensor cores
        for (int i = 0; i < BK / MMA_WIDTH; i++) {
            for (int j = 0; j < WM; j++) {
                wmma::load_matrix_sync(a_frag[j], SA + warpX * MMA_WIDTH * WM * BK + j * MMA_WIDTH * BK + i * MMA_WIDTH, BK);
            }
            for (int j = 0; j < WN; j++) {
                wmma::load_matrix_sync(b_frag[j], SB + warpY * WN * MMA_WIDTH + j * MMA_WIDTH + i * MMA_WIDTH * BN, BN);
            }
            for (int j = 0; j < WM; j++) {
                for (int k = 0; k < WN; k++) {
                    wmma::mma_sync(c_frag[j * WN + k], a_frag[j], b_frag[k], c_frag[j * WN + k]);
                }
            }
        }

        for (int i = 0; i < WM; i++) {
            for (int j = 0; j < WN; j++) {
                int row = i + warpX * WM;
                int col = j + warpY * WN;
                wmma::store_matrix_sync(SC + (row * BN + col) * MMA_WIDTH, c_frag[i * WN + j], BN, wmma::mem_row_major);
            }
        }
    }

    // sync threads to ensure all warps have stored their part
    __syncthreads();

    // finally store back to global
    for (int i = tid; i < BM * BN; i+= blockDim.x) {
        int blockRow = i / BN, blockCol = i % BN;
        int globalRow = blockRow + blockX * BM;
        int globalCol = blockCol + blockY * BN;
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = __float2half(beta * __half2float(C[globalRow * N + globalCol]) + alpha * SC[i]);
        }
    }
}

// A, B, and C are device pointers
void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int numBlocksM = (M + BM - 1) / BM;
    int numBlocksN = (N + BN - 1) / BN;
    int shmemNeeded = (BM + BN) * BK * sizeof(half) + (BM * BN) * sizeof(float);
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98*1024);
    matmul_kernel<<<numBlocksM * numBlocksN, 32 * NUM_WARPS, shmemNeeded>>>(A, B, C, M, N, K, alpha, beta);
}
