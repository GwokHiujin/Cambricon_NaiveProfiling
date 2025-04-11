#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int N, const int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Each thread computes one element of the block sub-matrix
    float sum = 0.0f;
    
    // Loop over all sub-matrices of A and B required to compute block sub-matrix
    for (int m = 0; m < K; m += 16) {
        // Load sub-matrices from global memory to shared memory
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];
        
        if ((blockRow * 16 + row < M) && (m + col < K))
            As[row][col] = A[(blockRow * 16 + row) * K + m + col];
        else
            As[row][col] = 0.0f;
            
        if ((m + row < K) && (blockCol * 16 + col < N))
            Bs[row][col] = B[(m + row) * N + blockCol * 16 + col];
        else
            Bs[row][col] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < 16; k++)
            sum += As[row][k] * Bs[k][col];
            
        __syncthreads();
    }
    
    // Write result to global memory
    if ((blockRow * 16 + row < M) && (blockCol * 16 + col < N))
        C[(blockRow * 16 + row) * N + blockCol * 16 + col] = sum;
}
