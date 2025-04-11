#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels, 
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_h,
    const int out_w
) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    const int total_elements = batch_size * out_channels * out_h * out_w;
    
    for (int idx = thread_pos; idx < total_elements; idx += total_threads) {
        const int w_out = idx % out_w;
        const int h_out = (idx / out_w) % out_h;
        const int c_out = (idx / (out_w * out_h)) % out_channels;
        const int b = idx / (out_w * out_h * out_channels);
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int h_in = (h_out + pad_h - kh) / stride_h;
                    const int w_in = (w_out + pad_w - kw) / stride_w;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        if ((h_out + pad_h - kh) % stride_h == 0 && 
                            (w_out + pad_w - kw) % stride_w == 0) {
                            
                            const int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                            const int weight_idx = ((c_in * out_channels + c_out) * kernel_h + kh) * kernel_w + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        output[idx] = sum;
    }
}
