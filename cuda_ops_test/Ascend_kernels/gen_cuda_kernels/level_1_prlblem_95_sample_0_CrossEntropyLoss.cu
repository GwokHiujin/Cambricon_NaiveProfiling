#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size)
        return;

    // Compute loss for sample i

    // find max_pred for numerical stability
    float max_pred = -INFINITY;
    for (int j = 0; j < num_classes; ++j) {
        float pred = predictions[i * num_classes + j];
        if (pred > max_pred) {
            max_pred = pred;
        }
    }

    // Compute log_sum_exp
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += expf(predictions[i * num_classes + j] - max_pred);
    }
    float log_sum_exp = logf(sum_exp);

    int target_class = targets[i];

    float loss_i = - (predictions[i * num_classes + target_class] - max_pred) + log_sum_exp;
    losses[i] = loss_i;
}
