#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int N, const int K) {

