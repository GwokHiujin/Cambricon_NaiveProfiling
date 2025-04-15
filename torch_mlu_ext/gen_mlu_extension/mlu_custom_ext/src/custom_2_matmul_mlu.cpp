#include "custom_2_matmul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor matmul_2_mlu(torch::Tensor A, torch::Tensor B) {
    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    auto B_contiguous = torch_mlu::cnnl_contiguous(B);
    auto B_impl = getMluTensorImpl(B_contiguous);
    auto B_ptr = B_impl->mlu_data_ptr();
    
    const int M = A_contiguous.size(0);
    const int K = A_contiguous.size(1);
    const int N = B_contiguous.size(1);
    
    auto C = at::zeros({M, N}, A_contiguous.options());
    
    // dim3 threadsPerBlock(16, 16);
    // dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    auto elem_num = A_contiguous.numel();
    matmul_2_kernel_entry(
    reinterpret_cast<float*>(A_ptr),
    reinterpret_cast<float*>(B_ptr),
    reinterpret_cast<float*>(C_ptr),
    M, N, K
    , elem_num);
    
    return C;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("matmul_2_mlu(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::matmul_2_mlu"),
        TORCH_FN(matmul_2_mlu));
}
    