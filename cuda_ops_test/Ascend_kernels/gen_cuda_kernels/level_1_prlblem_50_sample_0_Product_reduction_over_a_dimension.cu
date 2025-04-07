#include <cuda_runtime.h>
__global__ void product_reduction_kernel(const float* input, float* output, 
                                       int batch_size, int dim1, int dim2, int reduction_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reduction_dim == 1) {
        // Reduce over dim1
        int batch_idx = tid / dim2;
        int d2_idx = tid % dim2;
        
        if (batch_idx < batch_size && d2_idx < dim2) {
            float prod = 1.0f;
            for (int d1_idx = 0; d1_idx < dim1; d1_idx++) {
                prod *= input[batch_idx * dim1 * dim2 + d1_idx * dim2 + d2_idx];
            }
            output[batch_idx * dim2 + d2_idx] = prod;
        }
    }
}
