#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes) {

