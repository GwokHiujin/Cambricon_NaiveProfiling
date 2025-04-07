#include "custom_conv_transpose2d_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor conv_transpose2d_mlu(
    torch::Tensor input,
    torch::Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding
) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    auto weight_contiguous = torch_mlu::cnnl_contiguous(weight);
    auto weight_impl = getMluTensorImpl(weight_contiguous);
    auto weight_ptr = weight_impl->mlu_data_ptr();
    
    const int batch_size = input_contiguous.size(0);
    const int in_channels = input_contiguous.size(1);
    const int height = input_contiguous.size(2);
    const int width = input_contiguous.size(3);
    
    const int out_channels = weight_contiguous.size(1);
    const int kernel_h = weight_contiguous.size(2);
    const int kernel_w = weight_contiguous.size(3);
    
    const int stride_h = std::get<0>(stride);
    const int stride_w = std::get<1>(stride);
    const int pad_h = std::get<0>(padding);
    const int pad_w = std::get<1>(padding);
    
    const int out_h = (height - 1) * stride_h - 2 * pad_h + kernel_h;
    const int out_w = (width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = at::zeros({batch_size, out_channels, out_h, out_w}, input_contiguous.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_h * out_w + threads - 1) / threads;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    conv_transpose2d_kernel_entry(
    reinterpret_cast<float*>(input_ptr),
    reinterpret_cast<float*>(weight_ptr),
    reinterpret_cast<float*>(output_ptr),
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    out_h,
    out_w
    );
    
    return output;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("conv_transpose2d_mlu(
    Tensor input,
    Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding
) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::conv_transpose2d_mlu"),
            TORCH_FN(conv_transpose2d_mlu));
    }
    