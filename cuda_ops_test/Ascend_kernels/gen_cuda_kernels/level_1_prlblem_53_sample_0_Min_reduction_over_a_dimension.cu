#include <cuda_runtime.h>
#include <limits>
__global__ void min_reduction_kernel(const float* input, float* output, 
                                   int batch_size, int dim1, int dim2, int reduce_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reduce_dim == 1) {
        // Reduce over dim1
        if (idx < batch_size * dim2) {
            int batch_idx = idx / dim2;
            int d2_idx = idx % dim2;
            
            float min_val = std::numeric_limits<float>::max();
            for (int d1 = 0; d1 < dim1; d1++) {
                float val = input[batch_idx * dim1 * dim2 + d1 * dim2 + d2_idx];
                min_val = min(min_val, val);
            }
            output[idx] = min_val;
        }
    }
    else if (reduce_dim == 2) {
        // Reduce over dim2
        if (idx < batch_size * dim1) {
            int batch_idx = idx / dim1;
            int d1_idx = idx % dim1;
            
            float min_val = std::numeric_limits<float>::max();
            for (int d2 = 0; d2 < dim2; d2++) {
                float val = input[batch_idx * dim1 * dim2 + d1_idx * dim2 + d2];
                min_val = min(min_val, val);
            }
            output[idx] = min_val;
        }
    }
}
