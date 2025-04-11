#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* diag,
    const float* mat,
    float* out,
    const int N,
    const int M) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        out[row * M + col] = diag[row] * mat[row * M + col];
    }
}
