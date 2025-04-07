#include <cuda_runtime.h>

__global__ void log_softmax_kernel(const float* input, float* output, int batch_size, int dim) {

