#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool1d_cuda_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int batch_size, int channels, int input_length, int output_length,
                                      int kernel_size, int stride, int padding, int dilation) {

