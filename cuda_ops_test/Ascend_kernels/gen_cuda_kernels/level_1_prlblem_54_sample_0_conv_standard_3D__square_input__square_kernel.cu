#include <cuda_runtime.h>
__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int out_depth,
    int out_height,
    int out_width)
{
    int o_channel = blockIdx.x;
    int batch = blockIdx.y;
    int output_idx_flat = blockIdx.z * blockDim.x + threadIdx.x;

    int od = output_idx_flat / (out_height * out_width);
    int oh = (output_idx_flat / out_width) % out_height;
    int ow = output_idx_flat % out_width;

    if (od >= out_depth || oh >= out_height || ow >= out_width) return;

    float output_value = 0.0;

    for (int i_channel = 0; i_channel < in_channels; ++i_channel) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            int id = od * stride_d - padding_d + kd;
            if (id < 0 || id >= input_depth) continue;
            for (int kh = 0; kh < kernel_height; ++kh) {
                int ih = oh * stride_h - padding_h + kh;
                if (ih < 0 || ih >= input_height) continue;
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int iw = ow * stride_w - padding_w + kw;
                    if (iw < 0 || iw >= input_width) continue;
                    // Compute indices
                    int input_idx = ((batch * in_channels + i_channel) * input_depth + id) * input_height * input_width + ih * input_width + iw;
                    int weight_idx = ((o_channel * in_channels + i_channel) * kernel_depth + kd) * kernel_height * kernel_width + kh * kernel_width + kw;
                    output_value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((batch * out_channels + o_channel) * out_depth + od) * out_height * out_width + oh * out_width + ow;
    output[output_idx] = output_value;
}
