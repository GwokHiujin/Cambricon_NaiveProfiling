#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float *x, float *y, int N, int M) {

