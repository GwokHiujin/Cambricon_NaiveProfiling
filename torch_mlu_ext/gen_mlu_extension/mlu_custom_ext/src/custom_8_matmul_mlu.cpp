#include "custom_8_matmul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor matmul_5_mlu(torch::Tensor a, torch::Tensor b) {
    const torch_mlu::mlu::MLUGuard device_guard(a.device());
    auto a_contiguous = torch_mlu::cnnl_contiguous(a);
    auto a_impl = getMluTensorImpl(a_contiguous);
    auto a_ptr = a_impl->mlu_data_ptr();
    auto b_contiguous = torch_mlu::cnnl_contiguous(b);
    auto b_impl = getMluTensorImpl(b_contiguous);
    auto b_ptr = b_impl->mlu_data_ptr();
    
    const int M = a_contiguous.size(0);
    const int K = a_contiguous.size(1);
    const int N = b_contiguous.size(1);
    
    auto c = at::zeros({M, N}, a_contiguous.options());
    
    // dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
    // (M + TILE_SIZE - 1) / TILE_SIZE);
    
    auto c_contiguous = torch_mlu::cnnl_contiguous(c);
    auto c_impl = getMluTensorImpl(c_contiguous);
    auto c_ptr = c_impl->mlu_data_ptr();
    auto elem_num = a_contiguous.numel();
    matmul_5_kernel_entry(
    reinterpret_cast<float*>(a_ptr),
    reinterpret_cast<float*>(b_ptr),
    reinterpret_cast<float*>(c_ptr),
    M, N, K, elem_num);
    
    return c;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("matmul_5_mlu(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::matmul_5_mlu"),
        TORCH_FN(matmul_5_mlu));
}
    