#include <cuda_runtime.h>
__global__ void tensor_matrix_multiply_kernel(
    const float* __restrict__ A,    // (b, i, j, l)
    const float* __restrict__ B,    // (l, k)
    float* __restrict__ C,          // (b, i, j, k)
    int b, int i, int j, int l, int k)
{
    int total_batches = b * i * j;
    int batch_idx = blockIdx.x;

    if (batch_idx >= total_batches) return;

    int b_idx = batch_idx / (i * j);
    int ij_idx = batch_idx % (i * j);
    int i_idx = ij_idx / j;
    int j_idx = ij_idx % j;

    int k_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (k_idx >= k) return;

    // Index pointers
    int A_index = ((b_idx * i + i_idx) * j + j_idx) * l;
    int C_index = ((b_idx * i + i_idx) * j + j_idx) * k + k_idx;

    float sum = 0.0f;

    for (int l_idx = 0; l_idx < l; ++l_idx)
    {
        float a_val = A[A_index + l_idx];          // A[b_idx, i_idx, j_idx, l_idx]
        float b_val = B[l_idx * k + k_idx];        // B[l_idx, k_idx]
        sum += a_val * b_val;
    }

    C[C_index] = sum;
}
