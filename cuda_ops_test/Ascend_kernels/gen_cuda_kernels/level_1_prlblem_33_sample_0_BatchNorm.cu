#include <cuda_runtime.h>
__global__ void batch_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch_size * channels * height * width;
    
    if (idx < size) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        float x = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float shift = beta[c];
        
        output[idx] = scale * (x - mean) / sqrt(var + epsilon) + shift;
    }
}
