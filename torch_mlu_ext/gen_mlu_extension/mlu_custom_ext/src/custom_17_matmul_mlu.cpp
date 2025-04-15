#include "custom_17_matmul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor matmul_7_mlu(torch::Tensor A, torch::Tensor B_T) {
    int64_t M = 1024;
    int64_t K = 4096;
    int64_t N = 2048;

    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    auto B_T_contiguous = torch_mlu::cnnl_contiguous(B_T);
    auto B_T_impl = getMluTensorImpl(B_T_contiguous);
    auto B_T_ptr = B_T_impl->mlu_data_ptr();
    
    auto C = at::zeros({M, N}, A_contiguous.options());
    
    // dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    // dim3 grid_size((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    auto elem_num = A_contiguous.numel();
    matmul_7_kernel_entry(
    reinterpret_cast<float*>(A_ptr),
    reinterpret_cast<float*>(B_T_ptr),
    reinterpret_cast<float*>(C_ptr),
    M,
    N,
    K
    , elem_num);
    
    return C;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("matmul_7_mlu(Tensor A, Tensor B_T) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::matmul_7_mlu"),
        TORCH_FN(matmul_7_mlu));
}
    