#include "custom_47_sum_reduction_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor sum_reduction_mlu(torch::Tensor x, int64_t dim) {
    const torch_mlu::mlu::MLUGuard device_guard(x.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    auto sizes = x_contiguous.sizes();
    int64_t dim_size = sizes[dim];
    int64_t stride = 1;
    for (int64_t i = dim + 1; i < sizes.size(); ++i) {
    stride *= sizes[i];
    }
    int64_t num_elements = x_contiguous.numel() / dim_size;
    auto out_sizes = sizes.vec();
    out_sizes[dim] = 1;
    auto out = at::zeros(out_sizes, x_contiguous.options());
    
    const int64_t block_size = 256;
    const int64_t num_blocks = (num_elements + block_size - 1) / block_size;
    
    auto out_contiguous = torch_mlu::cnnl_contiguous(out);
    auto out_impl = getMluTensorImpl(out_contiguous);
    auto out_ptr = out_impl->mlu_data_ptr();
    auto elem_num = x_contiguous.numel();
    sum_reduction_kernel_entry(reinterpret_cast<float*>(x_ptr), reinterpret_cast<float*>(out_ptr), dim_size, stride, num_elements, elem_num);
    
    return out;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("sum_reduction_mlu(Tensor x, int64_t dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::sum_reduction_mlu"),
        TORCH_FN(sum_reduction_mlu));
}
    