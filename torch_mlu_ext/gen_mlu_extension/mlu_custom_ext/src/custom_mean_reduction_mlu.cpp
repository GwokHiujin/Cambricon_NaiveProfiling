#include "custom_mean_reduction_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor mean_reduction_mlu(torch::Tensor input, int dim) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    
    auto input_sizes = input_contiguous.sizes();
    int ndim = input_sizes.size();
    dim = dim < 0 ? dim + ndim : dim;
    
    int reduce_dim_size = input_sizes[dim];
    
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
    outer_size *= input_sizes[i];
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
    inner_size *= input_sizes[i];
    }
    
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
    if (i != dim) {
    output_sizes.push_back(input_sizes[i]);
    }
    }
    
    auto output = at::empty(output_sizes, input_contiguous.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    mean_reduction_kernel_entry(
    reinterpret_cast<float*>(input_ptr),
    reinterpret_cast<float*>(output_ptr),
    reduce_dim_size,
    outer_size,
    inner_size
    );
    
    return output;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("mean_reduction_mlu(Tensor input, int dim) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::mean_reduction_mlu"),
            TORCH_FN(mean_reduction_mlu));
    }
    