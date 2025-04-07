#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* input, int64_t* output, 
                            int batch_size, int dim1, int dim2) {

