#include <cuda_runtime.h>
#include <cuda.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Block size
    const int TILE_SIZE = 16;

    // Block row and column
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread row and column within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row index of C and A
    int row = by * TILE_SIZE + ty;
    // Column index of C and B
    int col = bx * TILE_SIZE + tx;

    // Accumulate result
    float value = 0.0f;

    // Loop over tiles of K dimension
    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Shared memory for A and B tiles
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load tile of A into shared memory
        if (row < M && (m * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + m * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if (col < N && (m * TILE_SIZE + ty) < K) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}
