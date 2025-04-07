#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool2d_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{

