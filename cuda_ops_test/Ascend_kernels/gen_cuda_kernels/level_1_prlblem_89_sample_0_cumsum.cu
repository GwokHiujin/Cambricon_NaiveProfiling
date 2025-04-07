#include <cuda_runtime.h>
__global__ void scan_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = tid / seq_len;
    int seq_id = tid % seq_len;
    
    if (batch_id >= batch_size) return;
    
    float sum = 0;
    for (int i = 0; i <= seq_id; i++) {
        sum += input[batch_id * seq_len + i];
    }
    output[tid] = sum;
}
