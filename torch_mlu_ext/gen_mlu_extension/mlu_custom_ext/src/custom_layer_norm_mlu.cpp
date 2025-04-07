#include "custom_layer_norm_mlu.h"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor layer_norm_mlu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    const torch_mlu::mlu::MLUGuard device_guard(predictions.device());
    auto input_contiguous = torch_mlu::cnnl_contiguous(input);
    auto input_impl = getMluTensorImpl(input_contiguous);
    auto input_ptr = input_impl->mlu_data_ptr();
    auto weight_contiguous = torch_mlu::cnnl_contiguous(weight);
    auto weight_impl = getMluTensorImpl(weight_contiguous);
    auto weight_ptr = weight_impl->mlu_data_ptr();
    auto bias_contiguous = torch_mlu::cnnl_contiguous(bias);
    auto bias_impl = getMluTensorImpl(bias_contiguous);
    auto bias_ptr = bias_impl->mlu_data_ptr();
    
    auto batch_size = input_contiguous.size(0);
    auto features = input_contiguous.size(1);
    auto dim1 = input_contiguous.size(2);
    auto dim2 = input_contiguous.size(3);
    auto output = at::zeros_like(input_contiguous);
    auto mean = at::zeros(features, input_contiguous.device());
    auto inv_var = at::zeros(features, input_contiguous.device());
    
    const int block_size = 256;
    const int num_blocks = (features + block_size - 1) / block_size;
    
    auto mean_contiguous = torch_mlu::cnnl_contiguous(mean);
    auto mean_impl = getMluTensorImpl(mean_contiguous);
    auto mean_ptr = mean_impl->mlu_data_ptr();
    auto mean_contiguous = torch_mlu::cnnl_contiguous(mean);
    auto mean_impl = getMluTensorImpl(mean_contiguous);
    auto mean_ptr = mean_impl->mlu_data_ptr();
    auto inv_var_contiguous = torch_mlu::cnnl_contiguous(inv_var);
    auto inv_var_impl = getMluTensorImpl(inv_var_contiguous);
    auto inv_var_ptr = inv_var_impl->mlu_data_ptr();
    auto output_contiguous = torch_mlu::cnnl_contiguous(output);
    auto output_impl = getMluTensorImpl(output_contiguous);
    auto output_ptr = output_impl->mlu_data_ptr();
    auto mean_contiguous = torch_mlu::cnnl_contiguous(mean);
    auto mean_impl = getMluTensorImpl(mean_contiguous);
    auto mean_ptr = mean_impl->mlu_data_ptr();
    auto inv_var_contiguous = torch_mlu::cnnl_contiguous(inv_var);
    auto inv_var_impl = getMluTensorImpl(inv_var_contiguous);
    auto inv_var_ptr = inv_var_impl->mlu_data_ptr();
    compute_mean_kernel_entry(reinterpret_cast<float*>(input_ptr), reinterpret_cast<float*>(mean_ptr), batch_size, features, dim1, dim2);
    compute_inv_var_kernel_entry(reinterpret_cast<float*>(input_ptr), reinterpret_cast<float*>(mean_ptr), reinterpret_cast<float*>(inv_var_ptr), batch_size, features, dim1, dim2);
    
    const int output_block_size = 256;
    const int output_num_blocks = (batch_size * features * dim1 * dim2 + output_block_size - 1) / output_block_size;
    
    layer_norm_kernel_entry(reinterpret_cast<float*>(input_ptr), reinterpret_cast<float*>(weight_ptr), reinterpret_cast<float*>(bias_ptr), reinterpret_cast<float*>(output_ptr), reinterpret_cast<float*>(mean_ptr), reinterpret_cast<float*>(inv_var_ptr), batch_size, features, dim1, dim2);
    
    return output;
    
}


    TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
        m.def("layer_norm_mlu(Tensor input, Tensor weight, Tensor bias) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
        m.impl(
            TORCH_SELECTIVE_NAME("mlu_custom_ext::layer_norm_mlu"),
            TORCH_FN(layer_norm_mlu));
    }
    