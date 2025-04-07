#include <cuda_runtime.h>
__global__ void frobenius_norm_kernel(const float* input, float* output, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / norm;
    }
}
__global__ void square_sum_kernel(const float* input, float* partial_sums, int size) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (gid < size) {
        float val = input[gid];
        sum += val * val;
        gid += blockDim.x * gridDim.x;
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}
