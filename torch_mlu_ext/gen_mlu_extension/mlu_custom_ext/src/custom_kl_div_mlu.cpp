#include "custom_kl_div_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor kl_div_mlu(torch::Tensor log_predictions, torch::Tensor targets) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto log_predictions_contiguous = torch_mlu::cnnl_contiguous(log_predictions);
    auto log_predictions_impl = getMluTensorImpl(log_predictions_contiguous);
    auto log_predictions_ptr = log_predictions_impl->mlu_data_ptr();
    auto targets_contiguous = torch_mlu::cnnl_contiguous(targets);
    auto targets_impl = getMluTensorImpl(targets_contiguous);
    auto targets_ptr = targets_impl->mlu_data_ptr();
    
    auto batch_size = log_predictions_contiguous.size(0);
    auto num_classes = log_predictions_contiguous.size(1);
    auto out = at::zeros_like(log_predictions_contiguous);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * num_classes + block_size - 1) / block_size;
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    kl_div_kernel_entry(reinterpret_cast<float*>(log_predictions_ptr), reinterpret_cast<float*>(targets_ptr), reinterpret_cast<float*>(out_ptr), batch_size, num_classes);
    
    return out.sum(1).mean();
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("kl_div_mlu(Tensor log_predictions, Tensor targets) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::kl_div_mlu"),
            TORCH_FN(kl_div_mlu));
    }
    