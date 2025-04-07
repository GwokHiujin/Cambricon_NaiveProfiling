#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void batched_matrix_multiply_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int batch_size,
    int M, int K, int N) {

