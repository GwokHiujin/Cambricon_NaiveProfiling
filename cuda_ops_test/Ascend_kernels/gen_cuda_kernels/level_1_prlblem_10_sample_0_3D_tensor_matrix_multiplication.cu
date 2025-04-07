#include <cuda_runtime.h>

// CUDA kernel for 3D tensor-matrix multiplication
__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int N, const int M, const int K, const int L) {

