#include "custom_smooth_l1_loss_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor smooth_l1_loss_mlu(torch::Tensor predictions, torch::Tensor targets) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto predictions_contiguous = torch_mlu::cnnl_contiguous(predictions);
    auto predictions_impl = getMluTensorImpl(predictions_contiguous);
    auto predictions_ptr = predictions_impl->mlu_data_ptr();
    auto targets_contiguous = torch_mlu::cnnl_contiguous(targets);
    auto targets_impl = getMluTensorImpl(targets_contiguous);
    auto targets_ptr = targets_impl->mlu_data_ptr();
    
    AT_ASSERTM(predictions_contiguous.dim() == 1 || predictions_contiguous.dim() == 2, "predictions_contiguous must be 1D or 2D");
    AT_ASSERTM(targets_contiguous.dim() == 1 || targets_contiguous.dim() == 2, "targets_contiguous must be 1D or 2D");
    AT_ASSERTM(predictions_contiguous.sizes() == targets_contiguous.sizes(), "predictions_contiguous and targets_contiguous must have the same size");
    AT_ASSERTM(predictions_contiguous.scalar_type() == at::ScalarType::Float, "predictions_contiguous must be float");
    AT_ASSERTM(targets_contiguous.scalar_type() == at::ScalarType::Float, "targets_contiguous must be float");
    
    auto size = predictions_contiguous.numel();
    auto output = at::zeros_like(predictions_contiguous);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    smooth_l1_loss_kernel_launcher_entry(reinterpret_cast<float*>(predictions_ptr), reinterpret_cast<float*>(targets_ptr), reinterpret_cast<float*>(output_ptr), size);
    
    return at::mean(output);
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("smooth_l1_loss_mlu(Tensor predictions, Tensor targets) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::smooth_l1_loss_mlu"),
            TORCH_FN(smooth_l1_loss_mlu));
    }
    