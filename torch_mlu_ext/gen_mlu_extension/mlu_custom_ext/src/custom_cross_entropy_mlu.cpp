#include "custom_cross_entropy_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor cross_entropy_mlu(torch::Tensor predictions, torch::Tensor targets) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto predictions_contiguous = torch_mlu::cnnl_contiguous(predictions);
    auto predictions_impl = getMluTensorImpl(predictions_contiguous);
    auto predictions_ptr = predictions_impl->mlu_data_ptr();
    auto targets_contiguous = torch_mlu::cnnl_contiguous(targets);
    auto targets_impl = getMluTensorImpl(targets_contiguous);
    auto targets_ptr = targets_impl->mlu_data_ptr();
    
    // predictions_contiguous: [batch_size, num_classes]
    // targets_contiguous: [batch_size]
    
    int batch_size = predictions_contiguous.size(0);
    int num_classes = predictions_contiguous.size(1);
    
    auto losses = at::empty({batch_size}, predictions_contiguous.options());
    
    // Launch kernel
    const int threads = 128;
    const int blocks = (batch_size + threads - 1) / threads;
    
    auto losses_contiguous = torch_mlu::cnnl_contiguous(losses);
    auto losses_impl = getMluTensorImpl(losses_contiguous);
    auto losses_ptr = losses_impl->mlu_data_ptr();
    cross_entropy_kernel_entry(
    reinterpret_cast<float*>(predictions_ptr),
    targets_contiguous.data_ptr<int64_t>(),
    reinterpret_cast<float*>(losses_ptr),
    batch_size,
    num_classes);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // Return mean loss
    return losses.mean();
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("cross_entropy_mlu(Tensor predictions, Tensor targets) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::cross_entropy_mlu"),
            TORCH_FN(cross_entropy_mlu));
    }
    