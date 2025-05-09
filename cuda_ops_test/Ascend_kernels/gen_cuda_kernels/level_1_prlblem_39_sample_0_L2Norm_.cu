#include <cuda_runtime.h>

__global__ void l2_normalize_kernel(const float* x, float* y, int batch_size, int dim) {
    int sample_idx = blockIdx.x;
    const float* x_sample = x + sample_idx * dim;
    float* y_sample = y + sample_idx * dim;

    extern __shared__ float sdata[];

    // Each thread computes partial sum of squares
    float partial_sum = 0.0f;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        float val = x_sample[idx];
        partial_sum += val * val;
    }

    // Each thread writes partial sum to shared memory
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // After reduction, sdata[0] contains the total sum
    float norm = sqrtf(sdata[0] + 1e-8f); // Adding small epsilon to prevent division by zero

    // Now, each thread normalizes its elements
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        y_sample[idx] = x_sample[idx] / norm;
    }
}
