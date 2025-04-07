#include <cuda_runtime.h>

__global__ void l1_norm_kernel(const float* x, float* out, int dim, int batch_size) {

