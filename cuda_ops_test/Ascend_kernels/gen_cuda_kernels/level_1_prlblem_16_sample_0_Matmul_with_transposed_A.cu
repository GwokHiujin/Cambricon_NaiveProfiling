#include <cuda_runtime.h>
#define TILE_SIZE 16
__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                             int M, int K, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_col = t * TILE_SIZE + threadIdx.x;
        int tiled_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && tiled_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tiled_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (tiled_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[tiled_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = value;
}
