// #include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// template <typename float>
__device__ float compute_softplus(const float x) {
    if (x > static_cast<float>(20.0)) {
        return x;
    } else if (x < static_cast<float>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

// template <typename float>
__global__ void softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float x = input[idx];
        output[idx] = compute_softplus(x);
    }
}

// torch::Tensor softplus_cuda_forward(torch::Tensor input) {
//     auto output = torch::empty_like(input);
//     const int size = input.numel();
//     const int threads = 512;
//     const int blocks = (size + threads - 1) / threads;
// 
//     AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
//         softplus_kernel<float><<<blocks, threads>>>(
//             input.data_ptr<float>(),
//             output.data_ptr<float>(),
//             size);
//     }));
// 
//     return output;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
// }
