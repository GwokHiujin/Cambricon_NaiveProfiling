#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matvec_mul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;  // Each block computes one row
    int tid = threadIdx.x;

    extern __shared__ float shared_sum[];  // Shared memory for partial sums

    // Each thread computes partial sum over parts of K
    float sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        sum += A[row * K + i] * B[i];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[row] = shared_sum[0];
    }
}
