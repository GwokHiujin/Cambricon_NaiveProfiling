#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* out, int size) {

