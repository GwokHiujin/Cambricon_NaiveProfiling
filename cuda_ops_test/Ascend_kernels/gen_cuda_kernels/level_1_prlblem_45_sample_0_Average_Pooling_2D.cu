#include <cuda.h>
#include <cuda_runtime.h>
__global__ void avg_pool2d_kernel(const float* input,
                                  float* output,
                                  int N, int C, int H_in, int W_in,
                                  int H_out, int W_out,
                                  int kernel_size, int stride, int padding)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int total = N * C * H_out * W_out;

    if (index < total) {
        int w_out_idx = index % W_out;
        int h_out_idx = (index / W_out) % H_out;
        int c_idx = (index / (W_out * H_out)) % C;
        int n_idx = index / (W_out * H_out * C);

        int h_start = h_out_idx * stride - padding;
        int w_start = w_out_idx * stride - padding;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        h_end = min(h_end, H_in);
        w_end = min(w_end, W_in);

        float sum = 0.0;
        int count = 0;
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = ((n_idx * C + c_idx) * H_in + h) * W_in + w;
                sum += input[input_idx];
                count += 1;
            }
        }
        output[index] = sum / count;
    }
}
