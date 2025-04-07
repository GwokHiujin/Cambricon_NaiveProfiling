#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int batch_size, int seq_len) {

