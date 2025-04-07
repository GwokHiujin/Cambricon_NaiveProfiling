#include <cuda_runtime.h>
__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int N, const int M, const int K, const int L) {

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Block size
    const int BLOCK_SIZE = 16;

    // Shared memory for tiling
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Global indices
    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;
    const int batch = bz;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile from A into shared memory
        if (row < M && (t * BLOCK_SIZE + tx) < K && batch < N) {
            As[ty][tx] = A[batch * M * K + row * K + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if ((t * BLOCK_SIZE + ty) < K && col < L) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * L + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < L && batch < N) {
        C[batch * M * L + row * L + col] = sum;
    }
}
