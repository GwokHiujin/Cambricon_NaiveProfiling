#include "custom_masked_cumsum_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor masked_cumsum_mlu(torch::Tensor x, torch::Tensor mask) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    auto mask_contiguous = torch_mlu::cnnl_contiguous(mask);
    auto mask_impl = getMluTensorImpl(mask_contiguous);
    auto mask_ptr = mask_impl->mlu_data_ptr();
    
    auto batch_size = x_contiguous.size(0);
    auto seq_len = x_contiguous.size(1);
    auto out = at::zeros_like(x_contiguous);
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * seq_len;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    masked_cumsum_kernel_entry(
    reinterpret_cast<float*>(x_ptr),
    mask_contiguous.data_ptr<bool>(),
    reinterpret_cast<float*>(out_ptr),
    batch_size,
    seq_len
    );
    
    return out;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("masked_cumsum_mlu(Tensor x, Tensor mask) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::masked_cumsum_mlu"),
            TORCH_FN(masked_cumsum_mlu));
    }
    