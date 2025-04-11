#include <cuda_runtime.h>

__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float margin, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pos_dist = 0.0f;
        float neg_dist = 0.0f;
        for (int i = 0; i < 4096; i++) {
            float tmp = anchor[idx * 4096 + i] - positive[idx * 4096 + i];
            pos_dist += tmp * tmp;
            tmp = anchor[idx * 4096 + i] - negative[idx * 4096 + i];
            neg_dist += tmp * tmp;
        }
        pos_dist = sqrtf(pos_dist);
        neg_dist = sqrtf(neg_dist);
        out[idx] = max(0.0f, margin + pos_dist - neg_dist);
    }
}
