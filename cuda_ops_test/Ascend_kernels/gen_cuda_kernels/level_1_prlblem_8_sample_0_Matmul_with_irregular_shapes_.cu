#include <cuda_runtime.h>
#define TILE_SIZE 32
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                             const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K-1)/TILE_SIZE + 1; ++t) {
        if (row < M && t*TILE_SIZE + tx < K)
            As[ty][tx] = A[row*K + t*TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t*TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t*TILE_SIZE + ty)*N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row*N + col] = sum;
    }
}
