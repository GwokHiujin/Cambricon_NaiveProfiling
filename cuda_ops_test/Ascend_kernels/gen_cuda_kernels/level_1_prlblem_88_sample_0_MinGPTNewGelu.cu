#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int size)
{

