#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}
