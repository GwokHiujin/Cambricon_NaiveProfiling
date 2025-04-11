#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int batch_size, int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / seq_len;
    int seq_idx = tid % seq_len;
    
    if (batch_idx < batch_size && seq_idx < seq_len) {
        float sum = 0.0f;
        int offset = batch_idx * seq_len;
        
        for (int i = 0; i <= seq_idx; i++) {
            if (mask[offset + i]) {
                sum += x[offset + i];
            }
        }
        out[offset + seq_idx] = sum;
    }
}
