#include <cuda_runtime.h>
__global__ void maxpool3d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels, 
    const int dim1,
    const int dim2,
    const int dim3,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int out_dim1,
    const int out_dim2,
    const int out_dim3
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    
    if (idx < total_elements) {
        const int out_pos = idx;
        const int out_z = out_pos % out_dim3;
        const int out_y = (out_pos / out_dim3) % out_dim2;
        const int out_x = (out_pos / (out_dim3 * out_dim2)) % out_dim1;
        const int c = (out_pos / (out_dim3 * out_dim2 * out_dim1)) % channels;
        const int b = out_pos / (out_dim3 * out_dim2 * out_dim1 * channels);

        float maxval = -1e38;
        
        const int start_x = out_x * stride - padding;
        const int start_y = out_y * stride - padding;
        const int start_z = out_z * stride - padding;
        
        for(int kx = 0; kx < kernel_size; kx++) {
            const int in_x = start_x + kx * dilation;
            if (in_x >= 0 && in_x < dim1) {
                for(int ky = 0; ky < kernel_size; ky++) {
                    const int in_y = start_y + ky * dilation;
                    if (in_y >= 0 && in_y < dim2) {
                        for(int kz = 0; kz < kernel_size; kz++) {
                            const int in_z = start_z + kz * dilation;
                            if (in_z >= 0 && in_z < dim3) {
                                const int in_idx = ((b * channels + c) * dim1 + in_x) * dim2 * dim3 + 
                                                 in_y * dim3 + in_z;
                                maxval = max(maxval, input[in_idx]);
                            }
                        }
                    }
                }
            }
        }
        output[out_pos] = maxval;
    }
}
