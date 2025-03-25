#include "solve.h"
#include <cuda_runtime.h>

union FloatIntUnion {
    float f;
    int i;
};

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        FloatIntUnion fiu;
        fiu.f = input[idx];
        fiu.i = (fiu.i & ~(fiu.i >> 31));
        output[idx] = fiu.f;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
