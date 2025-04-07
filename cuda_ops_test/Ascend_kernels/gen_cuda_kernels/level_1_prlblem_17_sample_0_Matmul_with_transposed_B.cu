#define TILE_WIDTH 16

__global__ void matmul_kernel(const float* A, const float* B_T, float* C, int M, int N, int K) {

