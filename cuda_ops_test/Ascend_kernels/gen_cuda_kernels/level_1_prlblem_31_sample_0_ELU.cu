#include <cuda_runtime.h>

__global__ void elu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, float alpha, int size) {

