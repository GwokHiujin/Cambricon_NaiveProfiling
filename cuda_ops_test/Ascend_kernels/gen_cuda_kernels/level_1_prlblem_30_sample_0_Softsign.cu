#include <cuda_runtime.h>
#include <math.h>
__global__ void softsign_kernel(const float* x, float* out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
    }
}
