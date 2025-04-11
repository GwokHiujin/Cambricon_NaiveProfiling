#include <cuda.h>
#include <cuda_runtime.h>

__global__ void selu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float xi = x[idx];
        const float alpha = 1.6732632423543772848170429916717f;
        const float scale = 1.0507009873554804934193349852946f;
        float result = scale * (xi > 0 ? xi : alpha * (expf(xi) - 1.0f));
        y[idx] = result;
    }
}
