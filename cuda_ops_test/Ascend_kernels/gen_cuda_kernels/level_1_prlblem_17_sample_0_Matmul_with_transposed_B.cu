#define TILE_WIDTH 16

__global__ void matmul_kernel(const float* A, const float* B_T, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // M dimension
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // N dimension

    float Cvalue = 0.0f;

    // Loop over the tiles of K dimension
    for (int t = 0; t < ( (K + TILE_WIDTH - 1) / TILE_WIDTH ); ++t) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile of B_T into shared memory
        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B_T[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product
        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * N + col] = Cvalue;
}
