#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels, 
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int out_padding_h,
    const int out_padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups
) {
    const int h_out = blockIdx.y;
    const int w_out = blockIdx.x;
    const int batch_idx = blockIdx.z / out_channels;
    const int out_ch = blockIdx.z % out_channels;
    
    if (h_out >= height_out || w_out >= width_out || batch_idx >= batch_size)
        return;

    float sum = 0.0f;
    const int in_ch_per_group = in_channels / groups;
    const int out_ch_per_group = out_channels / groups;
    const int group = out_ch / out_ch_per_group;
    
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            const int h_in = (h_out + padding_h - kh * dilation_h) / stride_h;
            const int w_in = (w_out + padding_w - kw * dilation_w) / stride_w;
            
            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                for (int in_ch = group * in_ch_per_group; 
                     in_ch < (group + 1) * in_ch_per_group; 
                     in_ch++) {
                    
                    const float input_val = input[
                        batch_idx * in_channels * height_in * width_in +
                        in_ch * height_in * width_in +
                        h_in * width_in +
                        w_in
                    ];
                    
                    const float weight_val = weight[
                        in_ch * out_ch_per_group * kernel_h * kernel_w +
                        (out_ch % out_ch_per_group) * kernel_h * kernel_w +
                        kh * kernel_w +
                        kw
                    ];
                    
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    output[
        batch_idx * out_channels * height_out * width_out +
        out_ch * height_out * width_out +
        h_out * width_out +
        w_out
    ] = sum;
}
