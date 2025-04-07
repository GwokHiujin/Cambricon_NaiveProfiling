#pragma once
#include <torch/extension.h>
torch::Tensor layer_norm_mlu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);