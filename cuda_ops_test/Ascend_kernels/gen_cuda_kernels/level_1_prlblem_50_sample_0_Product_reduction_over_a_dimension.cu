#include <cuda_runtime.h>

__global__ void product_reduction_kernel(const float* input, float* output, 
                                       int batch_size, int dim1, int dim2, int reduction_dim) {

