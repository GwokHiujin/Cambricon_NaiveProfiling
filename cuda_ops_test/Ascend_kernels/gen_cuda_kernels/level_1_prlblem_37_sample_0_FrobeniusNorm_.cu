#include <cuda_runtime.h>

__global__ void frobenius_norm_kernel(const float* input, float* output, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / norm;
    }
}
