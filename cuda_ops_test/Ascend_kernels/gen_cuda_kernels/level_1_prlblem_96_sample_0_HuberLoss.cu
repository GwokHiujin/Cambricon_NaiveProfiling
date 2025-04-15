__device__ float smooth_l1_loss_kernel(float x) {
  float absx = fabsf(x);
  if (absx < 1.0f) {
    return 0.5f * absx * absx;
  } else {
    return absx - 0.5f;
  }
}
__global__ void smooth_l1_loss_kernel_launcher(const float* predictions, const float* targets, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = smooth_l1_loss_kernel(predictions[i] - targets[i]);
  }
}
