#include "custom_100_HingeLoss.h"
#include "mlu/bang_100_HingeLoss.h"

#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor hinge_loss_mlu(torch::Tensor predictions, torch::Tensor targets) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());

    // Process predictions
    auto predictions_contiguous = torch_mlu::cnnl_contiguous(predictions);
    auto predictions_impl = getMluTensorImpl(predictions_contiguous);
    auto predictions_ptr = predictions_impl->mlu_data_ptr();

    // Process targets
    auto targets_contiguous = torch_mlu::cnnl_contiguous(targets);
    auto targets_impl = getMluTensorImpl(targets_contiguous);
    auto targets_ptr = targets_impl->mlu_data_ptr();
    
    int n = predictions_contiguous.numel();

    auto output = at::empty_like(predictions_contiguous);
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();


    bang_100_HingeLoss_entry(
        reinterpret_cast<float*>(predictions_ptr), 
        reinterpret_cast<float*>(targets_ptr), 
        reinterpret_cast<float*>(output_ptr), 
        n
    );

    // int threads = 256;
    // int blocks = (n + threads - 1) / threads;

    // hinge_loss_kernel<<<blocks, threads>>>(
    //     predictions.data_ptr<float>(),
    //     targets.data_ptr<float>(),
    //     output.data_ptr<float>(),
    //     n
    // );

    // Compute the mean of the output tensor
    auto mean = at::mean(output);
    return mean;
}

TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("hinge_loss_mlu(Tensor predictions, Tensor targets) -> Tensor");
  }

  TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::hinge_loss_mlu"),
        TORCH_FN(hinge_loss_mlu));
  }