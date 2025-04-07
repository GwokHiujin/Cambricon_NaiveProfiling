#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16
__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread computes one element of C
    float Cvalue = 0;

    // Thread row and column within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Row and column indices of the element in C
    int rowC = blockRow * BLOCK_SIZE + row;
    int colC = blockCol * BLOCK_SIZE + col;

    // Loop over the tiles of K dimension
    for (int m = 0; m < (K + BLOCK_SIZE -1)/BLOCK_SIZE; ++m) {
        // Shared memory for A and B tiles
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Global indices for A and B
        int rowA = rowC;
        int colA = m * BLOCK_SIZE + col;
        int rowB = m * BLOCK_SIZE + row;
        int colB = colC;

        // Load A and B into shared memory
        As[row][col] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
        Bs[row][col] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;

        __syncthreads();

        // Multiply the tiles together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }
        __syncthreads();
    }

    // Write the result to output if within bounds
    if (rowC < M && colC < N)
        C[rowC * N + colC] = Cvalue;
}
