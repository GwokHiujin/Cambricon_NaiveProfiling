#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 16

__global__ void tall_skinny_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < N; i += TILE_SIZE) {
        if (row < M && (i + tx) < N)
            As[ty][tx] = A[row * N + (i + tx)];
        else
            As[ty][tx] = 0.0f;
            
        if ((i + ty) < N && col < M)
            Bs[ty][tx] = B[(i + ty) * M + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}
