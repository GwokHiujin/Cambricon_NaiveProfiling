#include "custom_l2_normalize_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor l2_normalize_mlu(torch::Tensor x) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    int batch_size = x_contiguous.size(0);
    int dim = x_contiguous.size(1);
    auto y = at::empty_like(x_contiguous);
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem_size = threads * sizeof(float);
    
    auto y_contiguous = torch_mlu::cnnl_contiguous(y);
    auto y_impl = getMluTensorImpl(y_contiguous);
    auto y_ptr = y_impl->mlu_data_ptr();
    l2_normalize_kernel_entry(
    reinterpret_cast<float*>(x_ptr), reinterpret_cast<float*>(y_ptr), batch_size, dim
    );
    
    return y;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("l2_normalize_mlu(Tensor x) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::l2_normalize_mlu"),
            TORCH_FN(l2_normalize_mlu));
    }
    