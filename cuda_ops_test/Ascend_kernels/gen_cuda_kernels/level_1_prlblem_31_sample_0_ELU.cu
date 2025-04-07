#include <cuda_runtime.h>
__global__ void elu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * (expf(x) - 1.0f);
    }
}
