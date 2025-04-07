__global__ void relu_kernel(const float* input, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (input[i] > 0) ? input[i] : 0;
  }
}

