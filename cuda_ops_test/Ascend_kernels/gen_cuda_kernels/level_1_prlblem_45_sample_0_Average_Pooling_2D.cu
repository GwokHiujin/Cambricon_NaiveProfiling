#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(const float* input,
                                  float* output,
                                  int N, int C, int H_in, int W_in,
                                  int H_out, int W_out,
                                  int kernel_size, int stride, int padding)
{

