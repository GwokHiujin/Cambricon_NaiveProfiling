#include <cuda_runtime.h>
__global__ void mse_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        out[idx] = diff * diff;
    }
}
