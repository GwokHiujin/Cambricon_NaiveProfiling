#include <cuda_runtime.h>

__global__ void log_softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    
    // Compute max for numerical stability
    float max_val = -INFINITY;
    for(int i = 0; i < dim; i++) {
        float val = input[batch_idx * dim + i];
        max_val = max(max_val, val);
    }
    __syncthreads();
    
    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for(int i = 0; i < dim; i++) {
        sum += exp(input[batch_idx * dim + i] - max_val);
    }
    float log_sum = log(sum);
    __syncthreads();
    
    // Compute final output
    for(int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[batch_idx * dim + i] = input[batch_idx * dim + i] - max_val - log_sum;
    }
}
