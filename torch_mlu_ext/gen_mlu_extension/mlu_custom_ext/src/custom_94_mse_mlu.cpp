#include "custom_94_mse_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor mse_mlu(torch::Tensor predictions, torch::Tensor targets) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto predictions_contiguous = torch_mlu::cnnl_contiguous(predictions);
    auto predictions_impl = getMluTensorImpl(predictions_contiguous);
    auto predictions_ptr = predictions_impl->mlu_data_ptr();
    auto targets_contiguous = torch_mlu::cnnl_contiguous(targets);
    auto targets_impl = getMluTensorImpl(targets_contiguous);
    auto targets_ptr = targets_impl->mlu_data_ptr();
    
    auto size = predictions_contiguous.numel();
    auto out = at::zeros_like(predictions_contiguous);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    mse_kernel_entry(reinterpret_cast<float*>(predictions_ptr), reinterpret_cast<float*>(targets_ptr), reinterpret_cast<float*>(out_ptr), size);
    
    return at::mean(out);
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("mse_mlu(Tensor predictions, Tensor targets) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::mse_mlu"),
        TORCH_FN(mse_mlu));
}
    