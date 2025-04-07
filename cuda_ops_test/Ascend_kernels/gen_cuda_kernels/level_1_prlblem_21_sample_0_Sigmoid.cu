#include <cuda_runtime.h>

__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {

