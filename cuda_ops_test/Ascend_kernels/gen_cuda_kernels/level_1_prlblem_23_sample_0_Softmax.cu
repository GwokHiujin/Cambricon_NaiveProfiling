#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cfloat>

__global__ void softmax_kernel_batch(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {

