#pragma once
#include <torch/extension.h>
torch::Tensor mean_reduction_mlu(torch::Tensor input, int dim);