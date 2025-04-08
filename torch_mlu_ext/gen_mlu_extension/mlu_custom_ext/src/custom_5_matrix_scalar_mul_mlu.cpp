#include "custom_5_matrix_scalar_mul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor matrix_scalar_mul_mlu(torch::Tensor A, double s) {
    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    
    auto size = A_contiguous.numel();
    auto C = at::zeros_like(A_contiguous);
    
    const int64_t block_size = 256;
    const int64_t num_blocks = (size + block_size - 1) / block_size;
    
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    auto size = A_contiguous.numel();
    matrix_scalar_mul_kernel_entry(
    reinterpret_cast<float*>(A_ptr),
    s,
    reinterpret_cast<float*>(C_ptr),
    size
    , size);
    
    return C;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("matrix_scalar_mul_mlu(Tensor A, double s) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::matrix_scalar_mul_mlu"),
        TORCH_FN(matrix_scalar_mul_mlu));
}
    