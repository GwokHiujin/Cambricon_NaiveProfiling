#pragma once
#include <torch/extension.h>
torch::Tensor relu_mlu(torch::Tensor input);