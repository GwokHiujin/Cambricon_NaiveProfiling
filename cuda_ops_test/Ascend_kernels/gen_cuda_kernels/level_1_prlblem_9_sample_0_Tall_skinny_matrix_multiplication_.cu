#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 16

__global__ void tall_skinny_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N) {

