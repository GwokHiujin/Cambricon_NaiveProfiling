#include <cuda_runtime.h>
#include <limits>

__global__ void min_reduction_kernel(const float* input, float* output, 
                                   int batch_size, int dim1, int dim2, int reduce_dim) {

