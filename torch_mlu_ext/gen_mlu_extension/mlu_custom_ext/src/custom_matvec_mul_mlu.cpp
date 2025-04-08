#include "custom_matvec_mul_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor matvec_mul_mlu(torch::Tensor A, torch::Tensor B) {
    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    auto B_contiguous = torch_mlu::cnnl_contiguous(B);
    auto B_impl = getMluTensorImpl(B_contiguous);
    auto B_ptr = B_impl->mlu_data_ptr();
    
    // Check device
    TORCH_CHECK(A_contiguous.is_cuda(), "A_contiguous must be a CUDA tensor");
    TORCH_CHECK(B_contiguous.is_cuda(), "B_contiguous must be a CUDA tensor");
    TORCH_CHECK(A_contiguous.is_contiguous(), "A_contiguous must be contiguous");
    TORCH_CHECK(B_contiguous.is_contiguous(), "B_contiguous must be contiguous");
    
    // Check dimensions
    int64_t M = A_contiguous.size(0);
    int64_t K = A_contiguous.size(1);
    TORCH_CHECK(B_contiguous.size(0) == K && B_contiguous.size(1) == 1, "B_contiguous must have size K x 1");
    
    // Allocate output tensor
    auto C = at::empty({M, 1}, A_contiguous.options());
    
    // Launch kernel
    int64_t threads = 256;
    int64_t blocks = M;
    size_t shared_mem_size = threads * sizeof(float);
    
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    matvec_mul_kernel_entry(
    reinterpret_cast<float*>(A_ptr), reinterpret_cast<float*>(B_ptr), reinterpret_cast<float*>(C_ptr), M, K);
    
    return C;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("matvec_mul_mlu(Tensor A, Tensor B) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::matvec_mul_mlu"),
            TORCH_FN(matvec_mul_mlu));
    }
    