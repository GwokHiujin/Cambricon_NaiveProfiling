#include "custom_6_matmul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor matmul_3_mlu(torch::Tensor A, torch::Tensor B) {
    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    auto B_contiguous = torch_mlu::cnnl_contiguous(B);
    auto B_impl = getMluTensorImpl(B_contiguous);
    auto B_ptr = B_impl->mlu_data_ptr();
    
    const auto M = A_contiguous.size(0);
    const auto K = A_contiguous.size(1);
    const auto N = B_contiguous.size(1);
    
    auto C = at::zeros({M, N}, at::device(A_contiguous.device()).dtype(A_contiguous.dtype()));
    
    // const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // const dim3 grid((N + BLOCK_SIZE -1)/BLOCK_SIZE, (M + BLOCK_SIZE -1)/BLOCK_SIZE);
    
    // Launch the kernel
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    auto elem_num = A_contiguous.numel();
    MatMulKernel_3_entry(reinterpret_cast<float*>(A_ptr), reinterpret_cast<float*>(B_ptr), reinterpret_cast<float*>(C_ptr), M, N, K, elem_num);
    
    return C;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("matmul_3_mlu(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::matmul_3_mlu"),
        TORCH_FN(matmul_3_mlu));
}
    