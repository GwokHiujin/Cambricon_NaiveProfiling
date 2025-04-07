#include <cuda.h>
#include <cuda_runtime.h>

__global__ void selu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N
) {

