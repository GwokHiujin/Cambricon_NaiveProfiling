#include <cuda_runtime.h>

__global__ void layer_norm_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                   const float* mean, const float* inv_var, int batch_size, int features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features * dim1 * dim2) {
        int feature_idx = (idx / (dim1 * dim2)) % features;
        output[idx] = (input[idx] - mean[feature_idx]) * inv_var[feature_idx] * weight[feature_idx] + bias[feature_idx];
    }
}

__global__ void compute_mean_kernel(const float* input, float* mean, int batch_size, int features, int dim1, int dim2) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size * dim1 * dim2; i++) {
            sum += input[feature_idx * batch_size * dim1 * dim2 + i];
        }
        mean[feature_idx] = sum / (batch_size * dim1 * dim2);
    }
}

__global__ void compute_inv_var_kernel(const float* input, const float* mean, float* inv_var, int batch_size, int features, int dim1, int dim2) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size * dim1 * dim2; i++) {
            float diff = input[feature_idx * batch_size * dim1 * dim2 + i] - mean[feature_idx];
            sum += diff * diff;
        }
        inv_var[feature_idx] = 1.0f / sqrt(sum / (batch_size * dim1 * dim2) + 1e-5f);
    }
}
