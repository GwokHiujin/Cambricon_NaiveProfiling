#include "custom_rms_norm_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor rms_norm_mlu(torch::Tensor x, float eps) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    const auto batch_size = x_contiguous.size(0);
    const auto num_features = x_contiguous.size(1);
    const auto dim1 = x_contiguous.size(2);
    const auto dim2 = x_contiguous.size(3);
    
    auto out = at::empty_like(x_contiguous);
    
    const int threads_per_block = num_features;
    const int blocks = batch_size * dim1 * dim2;
    
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    rms_norm_kernel_entry(
    reinterpret_cast<float*>(x_ptr),
    reinterpret_cast<float*>(out_ptr),
    batch_size,
    num_features,
    dim1,
    dim2,
    eps
    );
    
    return out;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("rms_norm_mlu(Tensor x, float eps) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::rms_norm_mlu"),
            TORCH_FN(rms_norm_mlu));
    }
    