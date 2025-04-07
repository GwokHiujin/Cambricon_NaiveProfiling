#include <cuda_runtime.h>
#include <math.h>

__global__ void softsign_kernel(const float* x, float* out, int size) {

