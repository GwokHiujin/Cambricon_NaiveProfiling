#include <cuda_runtime.h>
__global__ void matrix_scalar_mul_kernel(const float* A, float s, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}
