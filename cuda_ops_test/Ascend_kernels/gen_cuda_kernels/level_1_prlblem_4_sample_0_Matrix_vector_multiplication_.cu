#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matvec_mul_kernel(const float* A, const float* B, float* C, int M, int K) {

