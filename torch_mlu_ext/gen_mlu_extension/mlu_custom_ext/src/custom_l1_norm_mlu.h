#pragma once
#include <torch/extension.h>
torch::Tensor l1_norm_mlu(torch::Tensor x);