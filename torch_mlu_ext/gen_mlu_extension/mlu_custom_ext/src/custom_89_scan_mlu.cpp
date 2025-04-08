#include "custom_89_scan_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor scan_mlu(torch::Tensor input) {
    const torch_mlu::mlu::MLUGuard device_guard(input.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    auto batch_size = input_contiguous.size(0);
    auto seq_len = input_contiguous.size(1);
    auto output = at::zeros_like(input_contiguous);
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * seq_len;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    scan_kernel_entry(
    reinterpret_cast<float*>(input_ptr),
    reinterpret_cast<float*>(output_ptr),
    batch_size,
    seq_len
    );
    
    return output;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("scan_mlu(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::scan_mlu"),
        TORCH_FN(scan_mlu));
}
    