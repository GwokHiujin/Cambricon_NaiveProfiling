#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {

