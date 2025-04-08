#include "custom_maxpool1d_mlu_forward.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor maxpool1d_mlu_forward(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    const torch_mlu::mlu::MLUGuard device_guard(input.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    const int batch_size = input_contiguous.size(0);
    const int channels = input_contiguous.size(1);
    const int input_length = input_contiguous.size(2);
    
    int output_length = (input_length + 2 * padding - dilation * (kernel_size -1) -1) / stride +1;
    
    auto output = at::empty({batch_size, channels, output_length}, input_contiguous.options());
    
    const int threads = 1024;
    const int total_threads = batch_size * channels * output_length;
    const int blocks = (total_threads + threads -1)/threads;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    maxpool1d_cuda_kernel_entry(
    reinterpret_cast<float*>(input_ptr),
    reinterpret_cast<float*>(output_ptr),
    batch_size,
    channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation);
    return output;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("maxpool1d_mlu_forward(Tensor input, int kernel_size, int stride, int padding, int dilation) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::maxpool1d_mlu_forward"),
            TORCH_FN(maxpool1d_mlu_forward));
    }
    