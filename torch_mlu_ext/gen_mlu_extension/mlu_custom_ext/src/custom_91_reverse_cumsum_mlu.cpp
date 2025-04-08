#include "custom_91_reverse_cumsum_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor reverse_cumsum_mlu(torch::Tensor x) {
    const torch_mlu::mlu::MLUGuard device_guard(x.device());
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();
    
    auto N = x_contiguous.size(0);
    auto M = x_contiguous.size(1);
    auto y = at::zeros_like(x_contiguous);
    
    const int64_t threads = 1;  // one thread per block
    const int64_t blocks = N;
    
    auto y_contiguous = torch_mlu::cnnl_contiguous(y);
    auto y_impl = getMluTensorImpl(y_contiguous);
    auto y_ptr = y_impl->mlu_data_ptr();
    reverse_cumsum_kernel_entry(reinterpret_cast<float*>(x_ptr), reinterpret_cast<float*>(y_ptr), N, M);
    
    // Wait for the CUDA kernel to finish
    //cudaDeviceSynchronize();
    
    return y;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("reverse_cumsum_mlu(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::reverse_cumsum_mlu"),
        TORCH_FN(reverse_cumsum_mlu));
}
    