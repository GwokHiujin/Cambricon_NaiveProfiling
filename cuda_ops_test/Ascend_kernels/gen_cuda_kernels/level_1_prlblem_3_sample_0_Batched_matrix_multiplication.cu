#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
__global__ void batched_matrix_multiply_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int batch_size,
    int M, int K, int N) {

    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (batch < batch_size && row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a_element = A[batch * M * K + row * K + i];
            float b_element = B[batch * K * N + i * N + col];
            value += a_element * b_element;
        }
        C[batch * M * N + row * N + col] = value;
    }
}
