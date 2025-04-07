#include <cuda_runtime.h>
__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int n = batch_size * out_channels * depth_out * height_out * width_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    int w_out = index % width_out;
    int h_index = index / width_out;
    int h_out = h_index % height_out;
    int d_index = h_index / height_out;
    int d_out = d_index % depth_out;
    int c_index = d_index / depth_out;
    int c_out = c_index % out_channels;
    int batch = c_index / out_channels;

    int g = c_out / (out_channels / groups); // Group index
    int in_c_start = g * (in_channels / groups);
    int in_c_end = in_c_start + (in_channels / groups);

    float value = 0.0f;

    for (int c_in = in_c_start; c_in < in_c_end; ++c_in) {
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            int z_in = d_out * stride_d - padding_d + k_d * dilation_d;
            if (z_in < 0 || z_in >= depth_in) continue;
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                int y_in = h_out * stride_h - padding_h + k_h * dilation_h;
                if (y_in < 0 || y_in >= height_in) continue;
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    int x_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    if (x_in < 0 || x_in >= width_in) continue;

                    int input_idx = ((batch * in_channels + c_in) * depth_in + z_in) * height_in * width_in + y_in * width_in + x_in;
                    int weight_idx = ((c_out * (in_channels / groups) + (c_in - in_c_start)) * kernel_d + k_d) * kernel_h * kernel_w + k_h * kernel_w + k_w;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != NULL) {
        value += bias[c_out];
    }

    int output_idx = ((batch * out_channels + c_out) * depth_out + d_out) * height_out * width_out + h_out * width_out + w_out;
    output[output_idx] = value;
}
