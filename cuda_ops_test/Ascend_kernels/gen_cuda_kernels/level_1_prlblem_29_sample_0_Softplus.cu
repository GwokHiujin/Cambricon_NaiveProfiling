#include <cuda_runtime.h>
#include <cmath>
__global__ void softplus_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Softplus implementation: log(1 + exp(x))
        // Using numerically stable version to avoid overflow
        const float threshold = 20.0f;
        float x = input[idx];
        if (x > threshold) {
            output[idx] = x;  // For large x, softplus(x) ≈ x
        } else {
            output[idx] = logf(1.0f + expf(x));
        }
    }
}
