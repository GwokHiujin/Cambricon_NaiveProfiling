#include "custom_13_symmetric_matmul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor symmetric_matmul_mlu(torch::Tensor A, torch::Tensor B) {
    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    auto B_contiguous = torch_mlu::cnnl_contiguous(B);
    auto B_impl = getMluTensorImpl(B_contiguous);
    auto B_ptr = B_impl->mlu_data_ptr();
    
    const int N = A_contiguous.size(0);
    auto C = at::zeros({N, N}, A_contiguous.options());
    
    const int BLOCK_SIZE = 32;
    // dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    auto elem_num = A_contiguous.numel();
    symmetric_matmul_kernel_entry(
    reinterpret_cast<float*>(A_ptr),
    reinterpret_cast<float*>(B_ptr),
    reinterpret_cast<float*>(C_ptr),
    N
    , elem_num);
    
    return C;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("symmetric_matmul_mlu(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::symmetric_matmul_mlu"),
        TORCH_FN(symmetric_matmul_mlu));
}
    