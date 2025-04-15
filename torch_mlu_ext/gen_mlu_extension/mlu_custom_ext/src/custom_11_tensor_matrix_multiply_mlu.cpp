#include "custom_11_tensor_matrix_multiply_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor tensor_matrix_multiply_mlu(torch::Tensor A, torch::Tensor B) {
    const torch_mlu::mlu::MLUGuard device_guard(A.device());
    auto A_contiguous = torch_mlu::cnnl_contiguous(A);
    auto A_impl = getMluTensorImpl(A_contiguous);
    auto A_ptr = A_impl->mlu_data_ptr();
    auto B_contiguous = torch_mlu::cnnl_contiguous(B);
    auto B_impl = getMluTensorImpl(B_contiguous);
    auto B_ptr = B_impl->mlu_data_ptr();
    
    int b = A_contiguous.size(0);
    int i = A_contiguous.size(1);
    int j = A_contiguous.size(2);
    int l = A_contiguous.size(3);
    int k = B_contiguous.size(1);
    
    //TORCH_CHECK(B_contiguous.size(0) == l, "B_contiguous.size(0) must be equal to A_contiguous.size(3)");
    
    // Ensure A_contiguous and B_contiguous are contiguous and on the correct device
    A_contiguous = A_contiguous.contiguous();
    B_contiguous = B_contiguous.contiguous();
    
    auto C = at::zeros({b, i, j, k}, A_contiguous.options());
    
    int total_batches = b * i * j;
    
    int threads_per_block = 256;
    int blocks_per_k = (k + threads_per_block - 1) / threads_per_block;
    
    // Use a 2D grid: x dimension is total_batches, y dimension is blocks per k
    // dim3 grid_dim(total_batches, blocks_per_k);
    // dim3 block_dim(threads_per_block);
    
    // Launch kernel
    auto C_contiguous = torch_mlu::cnnl_contiguous(C);
    auto C_impl = getMluTensorImpl(C_contiguous);
    auto C_ptr = C_impl->mlu_data_ptr();
    auto elem_num = A_contiguous.numel();
    tensor_matrix_multiply_kernel_entry(
        reinterpret_cast<float*>(A_ptr), reinterpret_cast<float*>(B_ptr), reinterpret_cast<float*>(C_ptr), b, i, j, l, k, elem_num);
    
    // Synchronize to ensure completion
    //cudaDeviceSynchronize(elem_num);
    
    return C;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("tensor_matrix_multiply_mlu(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::tensor_matrix_multiply_mlu"),
        TORCH_FN(tensor_matrix_multiply_mlu));
}
    