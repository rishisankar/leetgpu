#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned int *image, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = image[i];
        // want to set 2nd, 3rd, 4th byte to 255 - b
        // 1st is left alone (not 4th, because little endian)
        // first byte (FB) = x & 0xFF000000
        // rest: x - FB
        // 0x00FFFFFF (255 in all but first byte)
        // Want: 0x00FFFFFF - (x-FB) + FB = 0x00FFFFFF - x + 2*FB
        image[i] = ((x & 0xFF000000) << 1) + 0x00FFFFFF - x;
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int N = width * height;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + 1023) / 1024;
    
    unsigned int *ptr = reinterpret_cast<unsigned int *>(image);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(ptr, N);
}
