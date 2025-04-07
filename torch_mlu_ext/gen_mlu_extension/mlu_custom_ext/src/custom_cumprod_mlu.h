#pragma once
#include <torch/extension.h>
torch::Tensor cumprod_mlu(torch::Tensor x, int64_t dim);