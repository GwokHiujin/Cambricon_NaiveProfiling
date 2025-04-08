#pragma once
#include <torch/extension.h>
torch::Tensor sum_reduction_mlu(torch::Tensor x, int64_t dim);