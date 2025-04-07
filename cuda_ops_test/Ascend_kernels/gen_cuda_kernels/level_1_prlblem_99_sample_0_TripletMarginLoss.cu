#include <cuda_runtime.h>

__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float margin, float* out, int size) {

