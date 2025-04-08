#include "custom_23_softmax_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor softmax_mlu(torch::Tensor input) {
    const torch_mlu::mlu::MLUGuard device_guard(input.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    // Ensure input_contiguous is contiguous and on CUDA
    auto input_contiguous = input_contiguous.contiguous();
    auto batch_size = input_contiguous.size(0);
    auto dim = input_contiguous.size(1);
    auto output = at::empty_like(input_contiguous);
    
    int64_t threads = 512; // Adjust as needed
    int64_t blocks = batch_size;
    
    size_t shared_mem_size = 2 * threads * sizeof(float);
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    auto size = input_contiguous.numel();
    softmax_kernel_batch_entry(reinterpret_cast<float*>(input_ptr), reinterpret_cast<float*>(output_ptr), batch_size, dim, size);
    
    return output;
    
}


TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
    m.def("softmax_mlu(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::softmax_mlu"),
        TORCH_FN(softmax_mlu));
}
    