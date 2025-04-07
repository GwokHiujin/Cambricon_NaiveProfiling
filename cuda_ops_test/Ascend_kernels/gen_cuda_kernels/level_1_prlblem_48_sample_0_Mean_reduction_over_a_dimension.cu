#include <cuda_runtime.h>
__global__ void mean_reduction_kernel(const float* input, float* output, 
                                    int reduce_dim_size, int outer_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outer_size * inner_size) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        
        float sum = 0.0f;
        for (int i = 0; i < reduce_dim_size; i++) {
            int input_idx = outer_idx * reduce_dim_size * inner_size + 
                           i * inner_size + inner_idx;
            sum += input[input_idx];
        }
        output[idx] = sum / reduce_dim_size;
    }
}
