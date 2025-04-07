#include <cuda_runtime.h>

__global__ void compute_mean_var_kernel(
    const float* __restrict__ x,
    float* mean,
    float* var,
    int N, int C, int H, int W) {
__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* y,
    float epsilon,
    int N, int C, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (index >= total_elements) return;

    int n = index / (C * H * W);
    int c = (index / (H * W)) % C;

    float mean_val = mean[n * C + c];
    float var_val = var[n * C + c];

    float gamma_c = gamma[c];
    float beta_c = beta[c];

    float x_val = x[index];
    float y_val = gamma_c * (x_val - mean_val) / sqrtf(var_val + epsilon) + beta_c;
    y[index] = y_val;
}

