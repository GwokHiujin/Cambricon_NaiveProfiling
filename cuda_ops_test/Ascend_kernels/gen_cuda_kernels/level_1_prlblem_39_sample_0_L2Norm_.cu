#include <cuda_runtime.h>

__global__ void l2_normalize_kernel(const float* x, float* y, int batch_size, int dim) {

