#include "custom_product_reduction_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor product_reduction_mlu(torch::Tensor input, int reduction_dim) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    auto batch_size = input_contiguous.size(0);
    auto dim1 = input_contiguous.size(1);
    auto dim2 = input_contiguous.size(2);
    
    auto options = at::TensorOptions()
    .dtype(input_contiguous.dtype())
    .device(input_contiguous.device());
    
    at::Tensor output;
    if (reduction_dim == 1) {
    output = at::empty({batch_size, dim2}, options);
    }
    
    const int threads = 256;
    const int blocks = (batch_size * dim2 + threads - 1) / threads;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    product_reduction_kernel_entry(
    reinterpret_cast<float*>(input_ptr),
    reinterpret_cast<float*>(output_ptr),
    batch_size, dim1, dim2, reduction_dim
    );
    
    return output;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("product_reduction_mlu(Tensor input, int reduction_dim) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::product_reduction_mlu"),
            TORCH_FN(product_reduction_mlu));
    }
    