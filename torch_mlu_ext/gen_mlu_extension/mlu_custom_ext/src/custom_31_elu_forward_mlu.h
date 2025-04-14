#pragma once
#include <torch/extension.h>
torch::Tensor elu_forward_mlu(torch::Tensor input);