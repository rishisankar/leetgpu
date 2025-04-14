#include "solve.h"
#include <cuda_runtime.h>

/*
Implementation of Flash Attention 2
https://arxiv.org/pdf/2307.08691

Q: Mxd
K: Nxd
V: Nxd
O: Mxd
l,m: Mx1
*/

constexpr int SRAM_SIZE = 4000;
// To store the vectors li and mi, we need at most
// d entries, which is at most 1024
constexpr int MAX_VECTOR_SIZE = 1024;
constexpr float NEGATIVE_INF = -1e20;

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * Load a block of the matrix from src to dst.
 * src intended to be global, dst intended to be shared memory.
 * Matrix is size MxN
 */
__device__ void matrix_block_load(
    float* dst, 
    const float* src, 
    int M,
    int N,
    int block_size,
    int block_idx
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = M * N;
    int block_start = block_idx * block_size * N;
    int block_end = block_start + block_size * N;
    for (int i = block_start + tid; i < block_end; i += num_threads) {
        dst[i - block_start] = (i < num_elts) ? src[i] : 0;
    }
}

/**
 * Load a block of the matrix from src to dst, and transpose it.
 * src intended to be global, dst intended to be shared memory.
 * Matrix is size MxN
 */
 __device__ void matrix_block_load_transpose(
    float* dst, // will be size loop_block_size x N but transposed
    const float* src, 
    int M,
    int N,
    int block_size,
    int loop_block_size,
    int block_idx
) {

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = M * N;
    int block_start = block_idx * block_size * N;
    int block_end = block_start + block_size * N;
    for (int i = block_start + tid; i < block_end; i += num_threads) {
        int r = (i - block_start) / N;
        int c = i % N;
        dst[c * loop_block_size + r] = (i < num_elts) ? src[i] : 0;
    }
}

/**
 * Store src into a block of dst.
 * src intended to be shared memory, dst intended to be global.
 * dst is size M x N, src is size block_size x N.
 */
__device__ void matrix_block_store(
    float* dst, 
    const float* src, 
    int M,
    int N,
    int block_size,
    int block_idx
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int block_start = block_idx * block_size * N;
    int block_end = min(M * N, block_start + block_size * N);
    for (int i = block_start + tid; i < block_end; i += num_threads) {
        dst[i] = src[i - block_start];
    }
}

/**
 * Fill array of size N with fill_value.
 */
__device__ void array_fill(
    float* array,
    float fill_value,
    int N
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < N; i += num_threads) {
        array[i] = fill_value;
    }
}

/**
 * Computes matrix multiplication A*BT.
 * A is of size MxK, B is of size NxK.
 * Output C is of size MxN.
 * If add_to_output, A*BT is added to C instead of overwriting it.
 * This is a simple version, not optimized for speed.
 */
template <bool add_to_output = false>
__device__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C, 
    int M,
    int N,
    int K
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = M * N;
    for (int i = tid; i < num_elts; i += num_threads) {
        int m = i / N;
        int n = i % N;
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[m * K + k] * B[n * K + k];
        }
        if constexpr (add_to_output) {
            C[i] += sum;
        } else {
            C[i] = sum;
        }
    }
}

__device__ void divide_by_scalar(
    float* array,
    float scalar,
    int N
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < N; i += num_threads) {
        array[i] /= scalar;
    }
}

/**
 * Assigns mi_cur to max(mi_prev, rowmax(Si)).
 * mi_cur / mi_prev are vectors of size Br in smem.
 * Si is a matrix of size Br x Bc in smem.
 */
__device__ void mi_update(
    float* mi_cur,
    const float* mi_prev,
    const float* Si,
    int Br,
    int Bc
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < Br; i += num_threads) {
        float max_val = mi_prev[i];
        for (int j = 0; j < Bc; ++j) {
            max_val = max(max_val, Si[i * Bc + j]);
        }
        mi_cur[i] = max_val;
    }
}

/**
 * Converts Si to Pi, where Pi = exp(Si - mi).
 * Si is a matrix of size Br x Bc in smem.
 * Pi is a matrix of size Br x Bc in smem.
 * mi is a vector of size Br in smem.
 */
__device__ void si_to_pi(
    float* SiPi,
    const float* mi,
    int Br,
    int Bc
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < Br * Bc; i += num_threads) {
        int r = i / Bc;
        SiPi[i] = exp(SiPi[i] - mi[r]);
    }
}

/**
 * Update li to exp(mi_prev - mi_cur) * li + rowsum(Pi).
 * li is a vector of size Br in smem.
 * Pi is a matrix of size Br x Bc in smem.
 * mi_prev is a vector of size Br in smem.
 * mi_cur is a vector of size Br in smem.
 */
__device__ void li_update(
    float* li,
    const float* Pi,
    const float* mi_prev,
    const float* mi_cur,
    int Br,
    int Bc
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < Br; i += num_threads) {
        float sum = 0;
        for (int j = 0; j < Bc; ++j) {
            sum += Pi[i * Bc + j];
        }
        li[i] = exp(mi_prev[i] - mi_cur[i]) * li[i] + sum;
    }
}

/**
 * Update Oi to diag(exp(mi_prev - mi_cur)) * Oi + Pi * V.
 * Oi is a matrix of size Br x d in smem.
 * Pi is a matrix of size Br x Bc in smem.
 * V is a matrix of size Bc x d in smem.
 * mi_prev, mi_cur are vectors of size Br in smem.
 */
__device__ void Oi_update(
    float* Oi,
    const float* Pi,
    const float* VT,
    const float* mi_prev,
    const float* mi_cur,
    int Br,
    int Bc,
    int d
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = Br * d;
    for (int i = tid; i < num_elts; i += num_threads) {
        int r = i / d;
        Oi[i] = exp(mi_prev[r] - mi_cur[r]) * Oi[i];
    }
    matrix_multiply<true>(Pi, VT, Oi, Br, d, Bc);
}

/**
 * Divide each row of Oi by that value of li.
 * Oi is a matrix of size Br x d in smem.
 * li is a vector of size Br in smem.
 */
__device__ void Oi_scale(
    float* Oi,
    const float* li,
    int Br,
    int d
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = Br * d;
    for (int i = tid; i < num_elts; i += num_threads) {
        int r = i / d;
        Oi[i] /= li[r];
    }
}

__global__ void flash_attention_2_kernel(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* O, 
    const int M, 
    const int N, 
    const int d, 
    const int Br, 
    const int Bc, 
    const int Tr, 
    const int Tc,
    const int alloc_size
) {
    extern __shared__ float s[];
    float *Oi = s;
    float *Qi = &s[alloc_size];
    // will first store Ki, then get overriden to ViT
    float *KiVi = &s[2 * alloc_size];
    // will first store Si, then get overriden to Pi
    float *SiPi = &s[3 * alloc_size];
    float *li = &s[4 * alloc_size];
    float *mi = &s[4 * alloc_size + MAX_VECTOR_SIZE];
    float *mi2 = &s[4 * alloc_size + 2 * MAX_VECTOR_SIZE];

    float* mi_prev = mi; // m(i,j-1)
    float* mi_cur = mi2; // m(i,j)

    for (int i = 0; i < Tr; i++) {
        int loopBr = min(Br, M - i * Br);
        matrix_block_load(Qi, Q, M, d, Br, i);
        array_fill(Oi, 0, loopBr * d);
        array_fill(li, 0, loopBr);
        array_fill(mi_prev, NEGATIVE_INF, loopBr);
        __syncthreads();
        for (int j = 0; j < Tc; j++) {
            int loopBc = min(Bc, N - j * Bc);
            matrix_block_load(KiVi, K, N, d, Bc, j);
            __syncthreads();
            matrix_multiply(Qi, KiVi, SiPi, loopBr, loopBc, d);
            __syncthreads();
            divide_by_scalar(SiPi, sqrtf(d), loopBr * loopBc);
            __syncthreads();
            mi_update(mi_cur, mi_prev, SiPi, loopBr, loopBc);
            __syncthreads();
            si_to_pi(SiPi, mi_cur, loopBr, loopBc);
            __syncthreads();
            li_update(li, SiPi, mi_prev, mi_cur, loopBr, loopBc);
            matrix_block_load_transpose(KiVi, V, N, d, Bc, loopBc, j);
            __syncthreads();
            Oi_update(Oi, SiPi, KiVi, mi_prev, mi_cur, loopBr, loopBc, d);
            __syncthreads();

            // swap mi_prev / mi_cur
            auto tmp = mi_prev;
            mi_prev = mi_cur;
            mi_cur = tmp;
        }
        Oi_scale(Oi, li, loopBr, d);
        __syncthreads();
        matrix_block_store(O, Oi, M, d, Br, i);
        __syncthreads();
    }
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int Bc = ceildiv(SRAM_SIZE, 4 * d);
    int Br = min(Bc, d);
    int Tr = ceildiv(M, Br);
    int Tc = ceildiv(N, Bc);

    int alloc_size = max(Br * Bc, Bc * d);
    int shmem_needed = (4 * alloc_size + 3 * MAX_VECTOR_SIZE) * sizeof(float);

    // call kernel
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = 1;
    flash_attention_2_kernel<<<blocksPerGrid, threadsPerBlock, shmem_needed>>>(
        Q, K, V, output, M, N, d, Br, Bc, Tr, Tc, alloc_size
    );
}
