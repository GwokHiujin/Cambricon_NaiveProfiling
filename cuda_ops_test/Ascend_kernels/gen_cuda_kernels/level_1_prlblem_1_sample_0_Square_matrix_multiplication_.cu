#include <cuda_runtime.h>
#define TILE_SIZE 32
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < N && (tile * TILE_SIZE + tx) < N)
            shared_A[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if (col < N && (tile * TILE_SIZE + ty) < N)
            shared_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
