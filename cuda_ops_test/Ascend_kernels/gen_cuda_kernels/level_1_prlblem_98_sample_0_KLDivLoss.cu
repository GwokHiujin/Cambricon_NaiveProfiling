#include <cuda_runtime.h>
#include <math.h>

__global__ void kl_div_kernel(const float* log_predictions, const float* targets, float* out, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        int b = idx / num_classes;
        int c = idx % num_classes;
        float log_p = log_predictions[idx];
        float target = targets[idx];
        out[idx] = target * (log(target) - log_p);
    }
}
