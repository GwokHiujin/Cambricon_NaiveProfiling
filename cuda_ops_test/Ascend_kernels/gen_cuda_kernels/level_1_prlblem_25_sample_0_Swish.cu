#include <cuda_runtime.h>

__global__ void swish_kernel(const float* input, float* output, int size) {

