#include "custom_softsign_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor softsign_mlu(torch::Tensor x) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    auto size = x_contiguous.numel();
    auto out = at::empty_like(x_contiguous);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    softsign_kernel_entry(reinterpret_cast<float*>(x_ptr), reinterpret_cast<float*>(out_ptr), size);
    
    return out;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("softsign_mlu(Tensor x) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::softsign_mlu"),
            TORCH_FN(softsign_mlu));
    }
    