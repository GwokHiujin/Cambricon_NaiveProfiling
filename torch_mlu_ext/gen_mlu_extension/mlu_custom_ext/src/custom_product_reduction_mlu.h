#pragma once
#include <torch/extension.h>
torch::Tensor product_reduction_mlu(torch::Tensor input, int64_t reduction_dim);