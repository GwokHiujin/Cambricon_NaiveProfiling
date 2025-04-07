#include <cuda_runtime.h>
#include <math.h>

__global__ void kl_div_kernel(const float* log_predictions, const float* targets, float* out, int batch_size, int num_classes) {

