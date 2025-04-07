#include <cuda.h>
#include <cuda_runtime.h>
__global__ void reverse_cumsum_kernel(const float *x, float *y, int N, int M) {
    int row = blockIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = M - 1; j >= 0; --j) {
            int idx = row * M + j;
            sum += x[idx];
            y[idx] = sum;
        }
    }
}
