#include <cuda_runtime.h>

// Optimized kernel for symmetric matrix multiplication
__global__ void symmetric_matmul_kernel(const float* A, const float* B, float* C, int N) {

