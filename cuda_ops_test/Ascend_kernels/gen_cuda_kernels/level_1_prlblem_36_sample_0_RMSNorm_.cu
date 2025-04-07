#include <cuda_runtime.h>
#include <vector>
__global__ void rms_norm_kernel(const float* __restrict__ x, float* __restrict__ out,
                                int batch_size, int num_features, int dim1, int dim2, float eps) {
    // Each block corresponds to one (b, i, j)
    // Threads correspond to c
    int idx = blockIdx.x; // block index
    int c = threadIdx.x;  // thread index

    if (idx >= batch_size * dim1 * dim2 || c >= num_features)
        return;

    int j = idx % dim2;
    int temp = idx / dim2;
    int i = temp % dim1;
    int b = temp / dim1;

    // Compute the index for x and out
    int x_idx = ((b * num_features + c) * dim1 + i) * dim2 + j;

    // Each thread computes x[b, c, i, j]^2
    float val = x[x_idx];
    float val_sq = val * val;

    // Use shared memory for reduction
    extern __shared__ float sdata[];

    sdata[c] = val_sq;
    __syncthreads();

    // Perform reduction to compute sum
    // Assuming num_features is power of 2
    for (unsigned int s = num_features / 2; s > 0; s >>= 1) {
        if (c < s) {
            sdata[c] += sdata[c + s];
        }
        __syncthreads();
    }

    float rms = 0.0f;
    if (c == 0) {
        float mean = sdata[0] / num_features;
        rms = sqrtf(mean + eps);
        sdata[0] = rms;
    }
    __syncthreads();

    rms = sdata[0];

    // Each thread normalizes its own x[b, c, i, j]
    out[x_idx] = val / rms;
}
