#include <cuda.h>
#include <cuda_runtime.h>

__global__ void lower_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {

