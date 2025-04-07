#include "custom_cumprod_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor cumprod_mlu(torch::Tensor x, int64_t dim) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    // Ensure input tensor is on CUDA and is contiguous
    TORCH_CHECK(x_contiguous.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x_contiguous.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x_contiguous.dim() == 2 && dim == 1, "Currently only supports 2D tensors with dim=1");
    
    int64_t batch_size = x_contiguous.size(0);
    int64_t length = x_contiguous.size(1);
    
    auto y = at::empty_like(x_contiguous);
    
    // Launch one kernel per batch
    auto y_contiguous = torch_mlu::cnnl_contiguous(y);
    auto y_impl = getMluTensorImpl(y_contiguous);
    auto y_ptr = y_impl->mlu_data_ptr();
    cumprod_kernel_entry(
    reinterpret_cast<float*>(x_ptr),
    reinterpret_cast<float*>(y_ptr),
    length
    );
    
    return y;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("cumprod_mlu(Tensor x, int64_t dim) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::cumprod_mlu"),
            TORCH_FN(cumprod_mlu));
    }
    