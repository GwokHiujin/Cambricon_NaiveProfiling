#include <cuda_runtime.h>
__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = (val + 3.0f) / 6.0f;
        val = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
        output[idx] = val;
    }
}
