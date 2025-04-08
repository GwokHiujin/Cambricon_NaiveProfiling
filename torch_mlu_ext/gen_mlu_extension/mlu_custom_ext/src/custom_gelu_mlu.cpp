#include "custom_gelu_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor gelu_mlu(torch::Tensor x) {
    const torch_mlu::mlu::MLUGuard device_guard(x.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    auto y = at::empty_like(x_contiguous);
    int64_t size = x_contiguous.numel();
    
    const int64_t threads = 256;
    const int64_t blocks = (size + threads - 1) / threads;
    
    auto y_contiguous = torch_mlu::cnnl_contiguous(y);
    auto y_impl = getMluTensorImpl(y_contiguous);
    auto y_ptr = y_impl->mlu_data_ptr();
    gelu_kernel_entry(reinterpret_cast<float*>(x_ptr), reinterpret_cast<float*>(y_ptr), size);
    
    return y;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("gelu_mlu(Tensor x) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::gelu_mlu"),
            TORCH_FN(gelu_mlu));
    }
    