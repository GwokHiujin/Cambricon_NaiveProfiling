#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                             int M, int K, int N) {

