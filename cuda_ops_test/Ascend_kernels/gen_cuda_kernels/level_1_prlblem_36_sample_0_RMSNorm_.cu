#include <cuda_runtime.h>
#include <vector>

__global__ void rms_norm_kernel(const float* __restrict__ x, float* __restrict__ out,
                                int batch_size, int num_features, int dim1, int dim2, float eps) {

