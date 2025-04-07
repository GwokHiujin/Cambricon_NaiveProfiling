#include <cuda_runtime.h>

// Tile sizes for shared memory
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                             const int M, const int N, const int K) {

