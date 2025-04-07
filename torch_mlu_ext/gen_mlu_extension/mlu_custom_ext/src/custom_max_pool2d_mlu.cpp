#include "custom_max_pool2d_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor max_pool2d_mlu(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    // Get input dimensions
    int batch_size = x_contiguous.size(0);
    int channels = x_contiguous.size(1);
    int input_height = x_contiguous.size(2);
    int input_width = x_contiguous.size(3);
    
    // Compute output dimensions
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width  = (input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Create output tensor
    auto options = at::TensorOptions().dtype(x_contiguous.dtype()).device(x_contiguous.device());
    auto output = at::empty({batch_size, channels, output_height, output_width}, options);
    
    // Launch CUDA kernel
    // Compute total number of threads
    int total_threads = batch_size * channels * output_height * output_width;
    const int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    max_pool2d_cuda_kernel_entry(
    reinterpret_cast<float*>(x_ptr),
    reinterpret_cast<float*>(output_ptr),
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    dilation
    );
    return output;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("max_pool2d_mlu(Tensor x, int kernel_size, int stride, int padding, int dilation) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::max_pool2d_mlu"),
            TORCH_FN(max_pool2d_mlu));
    }
    