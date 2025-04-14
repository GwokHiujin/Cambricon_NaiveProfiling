#pragma once
#include <torch/extension.h>
torch::Tensor product_reduction_mlu(torch::Tensor input);