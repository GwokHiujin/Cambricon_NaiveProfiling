#pragma once
#include <torch/extension.h>
torch::Tensor leaky_relu_mlu(torch::Tensor input, float negative_slope);