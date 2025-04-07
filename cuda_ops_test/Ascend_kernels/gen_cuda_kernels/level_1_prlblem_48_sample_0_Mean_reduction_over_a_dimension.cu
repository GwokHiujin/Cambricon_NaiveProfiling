#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(const float* input, float* output, 
                                    int reduce_dim_size, int outer_size, int inner_size) {

