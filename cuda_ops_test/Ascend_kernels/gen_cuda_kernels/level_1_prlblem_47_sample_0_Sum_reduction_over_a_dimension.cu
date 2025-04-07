#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* x, float* out, int dim_size, int stride, int num_elements) {

