#include "custom_99_triplet_margin_loss_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor triplet_margin_loss_mlu(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, double margin) {
    const torch_mlu::mlu::MLUGuard device_guard(anchor.device());
    auto anchor_contiguous = torch_mlu::cnnl_contiguous(anchor);
    auto anchor_impl = getMluTensorImpl(anchor_contiguous);
    auto anchor_ptr = anchor_impl->mlu_data_ptr();
    auto positive_contiguous = torch_mlu::cnnl_contiguous(positive);
    auto positive_impl = getMluTensorImpl(positive_contiguous);
    auto positive_ptr = positive_impl->mlu_data_ptr();
    auto negative_contiguous = torch_mlu::cnnl_contiguous(negative);
    auto negative_impl = getMluTensorImpl(negative_contiguous);
    auto negative_ptr = negative_impl->mlu_data_ptr();
    
    auto size = anchor_contiguous.size(0);
    auto out = at::zeros(size, at::TensorOptions().device(at::kCUDA));
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    triplet_margin_loss_kernel_entry(reinterpret_cast<float*>(anchor_ptr), reinterpret_cast<float*>(positive_ptr), reinterpret_cast<float*>(negative_ptr), margin, reinterpret_cast<float*>(out_ptr), size);
    
    return out.mean();
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("triplet_margin_loss_mlu(Tensor anchor, Tensor positive, Tensor negative, double margin) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::triplet_margin_loss_mlu"),
        TORCH_FN(triplet_margin_loss_mlu));
}
    