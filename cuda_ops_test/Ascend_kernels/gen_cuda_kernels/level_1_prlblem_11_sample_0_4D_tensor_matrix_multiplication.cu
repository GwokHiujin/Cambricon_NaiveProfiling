#include <cuda_runtime.h>

__global__ void tensor_matrix_multiply_kernel(
    const float* __restrict__ A,    // (b, i, j, l)
    const float* __restrict__ B,    // (l, k)
    float* __restrict__ C,          // (b, i, j, k)
    int b, int i, int j, int l, int k)
{

