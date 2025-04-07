#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 32

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int N) {

