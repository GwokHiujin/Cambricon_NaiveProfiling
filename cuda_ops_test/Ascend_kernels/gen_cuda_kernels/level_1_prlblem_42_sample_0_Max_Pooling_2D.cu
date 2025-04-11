#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool2d_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;
    if (index >= total_elements)
        return;

    // Compute n, c, h_out, w_out
    int w_out = index % output_width;
    int h_out = (index / output_width) % output_height;
    int c = (index / (output_width * output_height)) % channels;
    int n = index / (channels * output_height * output_width);

    // Compute h_start and w_start for the pooling window
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    float max_val = -FLT_MAX;
    // Iterate over the pooling window
    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            int h_in = h_start + i * dilation;
            int w_in = w_start + j * dilation;

            // Check if h_in and w_in are within input bounds
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width)
            {
                int input_index = n * channels * input_height * input_width
                                + c * input_height * input_width
                                + h_in * input_width
                                + w_in;
                float val = input[input_index];
                if (val > max_val)
                {
                    max_val = val;
                }
            }
        }
    }
    // Store the result
    int output_index = n * channels * output_height * output_width
                     + c * output_height * output_width
                     + h_out * output_width
                     + w_out;
    output[output_index] = max_val;
}
