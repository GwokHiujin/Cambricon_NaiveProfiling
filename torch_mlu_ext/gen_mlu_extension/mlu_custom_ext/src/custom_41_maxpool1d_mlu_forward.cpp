#include "custom_41_maxpool1d_mlu_forward.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor maxpool1d_mlu_forward(torch::Tensor input) {
    int64_t kernel_size = 4;
    int64_t stride = 2;
    int64_t padding = 2;
    int64_t dilation = 3;
    const torch_mlu::mlu::MLUGuard device_guard(input.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    const int64_t batch_size = input_contiguous.size(0);
    const int64_t channels = input_contiguous.size(1);
    const int64_t input_length = input_contiguous.size(2);
    
    int64_t output_length = (input_length + 2 * padding - dilation * (kernel_size -1) -1) / stride +1;
    
    auto output = at::empty({batch_size, channels, output_length}, input_contiguous.options());
    
    const int64_t threads = 1024;
    const int64_t total_threads = batch_size * channels * output_length;
    const int64_t blocks = (total_threads + threads -1)/threads;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    auto elem_num = input_contiguous.numel();
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
    dilation, elem_num);
    return output;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("maxpool1d_mlu_forward(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::maxpool1d_mlu_forward"),
        TORCH_FN(maxpool1d_mlu_forward));
}
    