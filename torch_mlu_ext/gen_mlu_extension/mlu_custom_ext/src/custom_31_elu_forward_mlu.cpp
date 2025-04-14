#include "custom_31_elu_forward_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor elu_forward_mlu(torch::Tensor input) {
    float alpha = 1.0;
    const torch_mlu::mlu::MLUGuard device_guard(input.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    int64_t size = input_contiguous.numel();
    auto output = at::empty_like(input_contiguous);
    
    const int64_t threads = 1024;
    const int64_t blocks = (size + threads - 1) / threads;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    auto elem_num = input_contiguous.numel();
    elu_forward_kernel_entry(reinterpret_cast<float*>(input_ptr), reinterpret_cast<float*>(output_ptr), alpha, size, elem_num);
    
    return output;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("elu_forward_mlu(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::elu_forward_mlu"),
        TORCH_FN(elu_forward_mlu));
}
    