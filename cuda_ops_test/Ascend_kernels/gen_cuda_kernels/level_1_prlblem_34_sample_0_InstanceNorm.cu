#include <cuda_runtime.h>

__global__ void compute_mean_var_kernel(
    const float* __restrict__ x,
    float* mean,
    float* var,
    int N, int C, int H, int W) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int HW = H * W;

    extern __shared__ float shared_data[];
    float* s_sum = shared_data;
    float* s_sum2 = shared_data + blockDim.x;

    float sum = 0.0f;
    float sum2 = 0.0f;

    int thread_idx = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = thread_idx; i < HW; i += num_threads) {
        int index = ((n * C + c) * H * W) + i;
        float val = x[index];
        sum += val;
        sum2 += val * val;
    }

    s_sum[thread_idx] = sum;
    s_sum2[thread_idx] = sum2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            s_sum[thread_idx] += s_sum[thread_idx + s];
            s_sum2[thread_idx] += s_sum2[thread_idx + s];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        float mean_val = s_sum[0] / HW;
        float var_val = s_sum2[0] / HW - mean_val * mean_val;
        mean[n * C + c] = mean_val;
        var[n * C + c] = var_val;
    }
}
