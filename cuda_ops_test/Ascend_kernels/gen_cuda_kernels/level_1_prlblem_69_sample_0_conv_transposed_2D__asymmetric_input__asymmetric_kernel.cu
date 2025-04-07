#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels, 
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int out_padding_h,
    const int out_padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups
) {

