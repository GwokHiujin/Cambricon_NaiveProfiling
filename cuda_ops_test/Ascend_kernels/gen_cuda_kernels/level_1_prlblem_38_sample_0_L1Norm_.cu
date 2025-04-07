#include <cuda_runtime.h>
__global__ void l1_norm_kernel(const float* x, float* out, int dim, int batch_size) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // Calculate the L1 norm for the current batch
    float l1_norm = 0.0;
    for (int i = thread_idx; i < dim; i += blockDim.x) {
        l1_norm += fabsf(x[batch_idx * dim + i]);
    }

    // Use shared memory to accumulate the L1 norm
    __shared__ float shared_l1_norm[256];
    shared_l1_norm[thread_idx] = l1_norm;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_l1_norm[thread_idx] += shared_l1_norm[thread_idx + stride];
        }
        __syncthreads();
    }

    // Normalize the input tensor
    l1_norm = shared_l1_norm[0];
    for (int i = thread_idx; i < dim; i += blockDim.x) {
        out[batch_idx * dim + i] = x[batch_idx * dim + i] / l1_norm;
    }
}
