#include <cuda_runtime.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define BLOCK_SIZE 16
__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0;

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int m = 0; m < numTiles; ++m) {

        int rowA = blockRow * BLOCK_SIZE + threadRow;
        int colA = m * BLOCK_SIZE + threadCol;

        if (rowA < M && colA < K) {
            Asub[threadRow][threadCol] = A[rowA * K + colA];
        } else {
            Asub[threadRow][threadCol] = 0.0f;
        }

        int rowB = m * BLOCK_SIZE + threadRow;
        int colB = blockCol * BLOCK_SIZE + threadCol;

        if (rowB < K && colB < N) {
            Bsub[threadRow][threadCol] = B[rowB * N + colB];
        } else {
            Bsub[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += Asub[threadRow][e] * Bsub[e][threadCol];
        }

        __syncthreads();
    }

    int rowC = blockRow * BLOCK_SIZE + threadRow;
    int colC = blockCol * BLOCK_SIZE + threadCol;

    if (rowC < M && colC < N) {
        C[rowC * N + colC] = Cvalue;
    }
}
