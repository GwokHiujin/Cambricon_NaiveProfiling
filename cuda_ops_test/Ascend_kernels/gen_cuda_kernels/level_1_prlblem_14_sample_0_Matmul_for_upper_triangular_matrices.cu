#include <cuda_runtime.h>

__global__ void triu_matmul_kernel(const float* A, const float* B, float* C, int N) {

