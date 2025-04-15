__global__ void cumprod_kernel(const float* x, float* y, int64_t length) {
    int batch_idx = blockIdx.x;
    const float* x_batch = x + batch_idx * length;
    float* y_batch = y + batch_idx * length;

    float cumprod = 1.0f;
    for (int64_t i = 0; i < length; ++i) {
        cumprod *= x_batch[i];
        y_batch[i] = cumprod;
    }
}
