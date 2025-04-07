#include <cuda_runtime.h>

__global__ void scan_kernel(const float* input, float* output, int batch_size, int seq_len) {

