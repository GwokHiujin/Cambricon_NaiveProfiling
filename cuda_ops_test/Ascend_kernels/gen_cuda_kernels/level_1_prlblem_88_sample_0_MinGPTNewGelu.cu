#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float xi = x[idx];
        float x3 = xi * xi * xi;
        const float c0 = 0.044715f;
        const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
        float tanharg = sqrt_2_over_pi * (xi + c0 * x3);
        float tanhres = tanhf(tanharg);
        y[idx] = 0.5f * xi * (1.0f + tanhres);
    }
}
