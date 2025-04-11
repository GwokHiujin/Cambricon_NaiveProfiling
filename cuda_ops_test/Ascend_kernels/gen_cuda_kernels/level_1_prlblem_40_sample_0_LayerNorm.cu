#include <cuda_runtime.h>

__global__ void layer_norm_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                   const float* mean, const float* inv_var, int batch_size, int features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features * dim1 * dim2) {
        int feature_idx = (idx / (dim1 * dim2)) % features;
        output[idx] = (input[idx] - mean[feature_idx]) * inv_var[feature_idx] * weight[feature_idx] + bias[feature_idx];
    }
}
