#include <cuda_runtime.h>
__global__ void argmin_kernel(const scalar_t* input, int64_t* output, 
                            int batch_size, int dim1, int dim2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * dim2) {
        int batch_idx = tid / dim2;
        int col_idx = tid % dim2;
        
        scalar_t min_val = input[batch_idx * dim1 * dim2 + col_idx];
        int min_idx = 0;
        
        for(int i = 1; i < dim1; i++) {
            scalar_t curr_val = input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
            if(curr_val < min_val) {
                min_val = curr_val;
                min_idx = i;
            }
        }
        output[batch_idx * dim2 + col_idx] = min_idx;
    }
}
