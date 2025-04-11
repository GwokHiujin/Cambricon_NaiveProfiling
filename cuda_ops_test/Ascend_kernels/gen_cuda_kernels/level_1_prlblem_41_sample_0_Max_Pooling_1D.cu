#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool1d_cuda_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int batch_size, int channels, int input_length, int output_length,
                                      int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_length) return;

    int out_pos = idx % output_length;
    int c = (idx / output_length) % channels;
    int n = idx / (channels * output_length);

    int in_start = out_pos * stride - padding;
    float max_val = -FLT_MAX;
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = in_start + k * dilation;
        if (in_pos >= 0 && in_pos < input_length) {
            int input_idx = n * channels * input_length + c * input_length + in_pos;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }
    int output_idx = n * channels * output_length + c * output_length + out_pos;
    output[output_idx] = max_val;
}
