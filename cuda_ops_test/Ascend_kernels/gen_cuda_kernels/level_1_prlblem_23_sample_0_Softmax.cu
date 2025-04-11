#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cfloat>

__global__ void softmax_kernel_batch(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {
    // Each block handles one batch
    int batch = blockIdx.x;
    if (batch < batch_size) {
        // Shared memory for max and sum
        extern __shared__ float sdata[];
        float* smax = sdata;
        float* ssum = sdata + blockDim.x;

        // Each thread handles index tid
        int tid = threadIdx.x;
        float max_val = -FLT_MAX;
        for (int i = tid; i < dim; i += blockDim.x) {
            float val = input[batch * dim + i];
            if (val > max_val)
                max_val = val;
        }
        // Reduce max_val across threads
        smax[threadIdx.x] = max_val;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
            if (threadIdx.x < s) {
                if (smax[threadIdx.x] < smax[threadIdx.x + s]) {
                    smax[threadIdx.x] = smax[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        max_val = smax[0];
        __syncthreads();

        // Now compute the sum of exp(x - max)
        float sum = 0.0f;
        for (int i = tid; i < dim; i += blockDim.x) {
            float val = __expf(input[batch * dim + i] - max_val);
            output[batch * dim + i] = val; // Temporarily store exp(x - max)
            sum += val;
        }
        // Reduce sum across threads
        ssum[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
            if (threadIdx.x < s) {
                ssum[threadIdx.x] += ssum[threadIdx.x + s];
            }
            __syncthreads();
        }
        sum = ssum[0];
        __syncthreads();

        // Finally compute output = exp(x - max) / sum
        for (int i = tid; i < dim; i += blockDim.x) {
            output[batch * dim + i] = output[batch * dim + i] / sum;
        }
    }
}
