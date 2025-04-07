#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
__global__ void argmax_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2
    ) {
    int n = blockIdx.x; // batch index

    int k = threadIdx.x + blockIdx.y * blockDim.x; // dim2 index
    if (k >= dim2) return;

    // Initialize max value and index
    float max_val = -FLT_MAX;
    int64_t max_idx = 0;

    for (int d = 0; d < dim1; d++) {
        int idx = n * dim1 * dim2 + d * dim2 + k;
        float val = input[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    // Write the index of the maximum value to output
    output[n * dim2 + k] = max_idx;
}
