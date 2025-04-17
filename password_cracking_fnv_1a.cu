#include "solve.h"
#include <cuda_runtime.h>

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ unsigned int fnv1a_hash_bytes(const unsigned char* data, int length) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    for (int i = 0; i < length; i++) {
        hash = (hash ^ data[i]) * FNV_PRIME;
    }
    return hash;
}

__device__ unsigned int repeated_hash(const unsigned char* password, int password_length, int R) {
    unsigned int hash = fnv1a_hash_bytes(password, password_length);
    for (int i = 0; i < R - 1; i++) {
        hash = fnv1a_hash_bytes((const unsigned char*)&hash, sizeof(hash));
    }
    return hash;
}

template<int password_length>
__global__ void password_crack_kernel(unsigned int target_hash, int R, char* output_password) {
    int tid = threadIdx.x;
    
    // generate password
    char test_pw[password_length + 1];
    test_pw[password_length] = '\0';
    test_pw[0] = 'a' + (tid % 26);
    if constexpr (password_length > 1) {
        test_pw[1] = 'a' + (tid / 26);
    }
    if constexpr (password_length > 2) {
        int block_id = blockIdx.x;
        #pragma unroll
        for (int i = 2; i < password_length; i++) {
            test_pw[i] = 'a' + (block_id % 26);
            block_id /= 26;
        }
    }

    unsigned int hash = repeated_hash((const unsigned char*)test_pw, password_length, R);

    if (hash == target_hash) {
        // found the password
        for (int i = 0; i < password_length; i++) {
            output_password[i] = test_pw[i];
        }
        output_password[password_length] = '\0';
    }
}


// output_password is a device pointer
void solve(unsigned int target_hash, int password_length, int R, char* output_password) {
    switch (password_length) {
        case 1:
            password_crack_kernel<1><<<1, 26>>>(target_hash, R, output_password);
            break;
        case 2:
            password_crack_kernel<2><<<1, 26*26>>>(target_hash, R, output_password);
            break;
        case 3:
            password_crack_kernel<3><<<26, 26*26>>>(target_hash, R, output_password);
            break;
        case 4:
            password_crack_kernel<4><<<26*26, 26*26>>>(target_hash, R, output_password);
            break;
        case 5:
            password_crack_kernel<5><<<26*26*26, 26*26>>>(target_hash, R, output_password);
            break;
        case 6:
            password_crack_kernel<6><<<26*26*26*26, 26*26>>>(target_hash, R, output_password);
            break;
        default:
            // can't get here
            return;
    }
}
