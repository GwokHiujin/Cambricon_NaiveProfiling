#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, int size) {

