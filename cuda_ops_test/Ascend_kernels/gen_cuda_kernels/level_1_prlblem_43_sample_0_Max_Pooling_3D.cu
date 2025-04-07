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

