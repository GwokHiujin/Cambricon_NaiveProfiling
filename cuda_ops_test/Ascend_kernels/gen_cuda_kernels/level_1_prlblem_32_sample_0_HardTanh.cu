#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {

