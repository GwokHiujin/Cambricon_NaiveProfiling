#pragma once
#include <torch/extension.h>
torch::Tensor new_gelu_mlu(torch::Tensor x);