#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {

