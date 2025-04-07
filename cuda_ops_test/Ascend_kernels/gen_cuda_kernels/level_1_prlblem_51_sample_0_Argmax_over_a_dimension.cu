#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2
    ) {

