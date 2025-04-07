#pragma once
#include <torch/extension.h>
torch::Tensor selu_forward_mlu(torch::Tensor x);