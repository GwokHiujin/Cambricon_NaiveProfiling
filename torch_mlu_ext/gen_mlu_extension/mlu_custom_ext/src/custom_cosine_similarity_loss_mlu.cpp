#include "custom_cosine_similarity_loss_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor cosine_similarity_loss_mlu(torch::Tensor predictions, torch::Tensor targets) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto predictions_contiguous = torch_mlu::cnnl_contiguous(predictions);
    auto predictions_impl = getMluTensorImpl(predictions_contiguous);
    auto predictions_ptr = predictions_impl->mlu_data_ptr();
    auto targets_contiguous = torch_mlu::cnnl_contiguous(targets);
    auto targets_impl = getMluTensorImpl(targets_contiguous);
    auto targets_ptr = targets_impl->mlu_data_ptr();
    
    const auto batch_size = predictions_contiguous.size(0);
    const auto input_size = predictions_contiguous.size(1);
    
    auto losses = at::empty({batch_size}, predictions_contiguous.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    const size_t shared_mem_size = threads * sizeof(float);
    
    auto losses_contiguous = torch_mlu::cnnl_contiguous(losses);
    auto losses_impl = getMluTensorImpl(losses_contiguous);
    auto losses_ptr = losses_impl->mlu_data_ptr();
    cosine_similarity_loss_kernel_entry(
    reinterpret_cast<float*>(predictions_ptr),
    reinterpret_cast<float*>(targets_ptr),
    reinterpret_cast<float*>(losses_ptr),
    batch_size,
    input_size
    );
    
    return losses;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("cosine_similarity_loss_mlu(Tensor predictions, Tensor targets) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::cosine_similarity_loss_mlu"),
            TORCH_FN(cosine_similarity_loss_mlu));
    }
    