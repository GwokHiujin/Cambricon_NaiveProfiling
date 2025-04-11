#include <cuda_runtime.h>

__global__ void max_reduction_kernel(const float* x, float* out, int dim_size, int stride, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int batch_idx = idx / stride;
        int element_idx = idx % stride;
        
        float max_val = -FLT_MAX;
        for (int i = 0; i < dim_size; ++i) {
            float val = x[batch_idx * dim_size * stride + i * stride + element_idx];
            if (val > max_val) {
                max_val = val;
            }
        }
        out[idx] = max_val;
    }
}
